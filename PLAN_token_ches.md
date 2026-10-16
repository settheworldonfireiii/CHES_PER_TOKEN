# Plan : CHES par token dans le repo officiel (NON IMPLÉMENTÉ)

Copie de `github.com/princeton-nlp/unintentional-unalignment` (commit `82e5dca`).
Même idée que pour AMPO, mais dans leur setup DPO : une paire (y⁺, y⁻) par exemple, avec un modèle de référence.

## Idée

Le terme du CHES qui fait baisser `log π(y⁺)` est le terme croisé, décomposé exactement par token de y⁻ :

    <S⁻, S⁺> = Σ_j <h⁻_j, S⁺>

On filtre donc les tokens de y⁻ qui ressemblent à la réponse positive entière :
`cos_j = <h⁻_j, S⁺> / (‖h⁻_j‖ · ‖S⁺‖ + 1e-6)`, retiré de `log π(y⁻)` si `cos_j > tau`. y⁺ n'est jamais filtré
(retirer un token de y⁺ lui enlèverait son gradient vers le haut). Par défaut `tau = 2.0` : rien n'est filtré.

Ici tout se fait en une seule passe : chaque paire est traitée dans le même `_step` (pas de groupe réparti sur
plusieurs GPU comme dans AMPO).

## Changements, fichier par fichier

| # | Fichier : ligne | Changement | Rôle |
|---|---|---|---|
| 1 | `common_dpo/trainers/dpo_trainer.py:146` (`__process_input_ids`) | `out = self.model(**input_data, output_hidden_states=True)` ; `logits = out.logits` ; garder `H = out.hidden_states[-1][:, :-1].detach().float()` et le renvoyer en plus | `H[:, t]` = état qui prédit le token t+1, aligné avec `logprobs` (`logits[:, :-1]`) et `get_prompt_mask` (`[:, :-1]`) ; ce sont les positions de `ches.py` |
| 2 | `dpo_trainer.py:216–217` (`__perform_dpo_or_ipo_step`) | récupérer `H_w`, `H_l` des deux appels | |
| 3 | après `mask_w`, `mask_l` (`:219–220`) | `S_w = (H_w * mask_w[..., None]).sum(1)` | somme de la réponse positive entière (seule `S_w` est nécessaire) |
| 4 | ligne suivante | `cos_l = (H_l @ S_w[..., None]).squeeze(-1) / (H_l.norm(dim=-1) * S_w.norm(dim=-1, keepdim=True) + 1e-6)` | `cos_j` pour chaque token de y⁻ |
| 5 | ligne suivante | `keep_l = mask_l * (cos_l <= self.config.token_ches_max_cos)` | masque filtré de y⁻ |
| 6 | `:223` et `:225` | remplacer `mask_l` par `keep_l` dans `seq_logprobs_l` **et** `seq_old_logprobs_l` | même masque pour la politique et la référence, sinon la marge DPO est biaisée ; `mask_w` reste inchangé |
| 7 | — | (supprimé : y⁺ n'est plus filtré) | |
| 8 | dans `stats` | `"train/token_ches_filtered_frac": 1 - keep_l.sum() / mask_l.sum()` | voir combien de tokens de y⁻ sont retirés |
| 9 | `common_dpo/trainers/dpo_config.py:77` | `token_ches_max_cos: float = 2.0` | champ de config ; 2.0 = désactivé |
| 10 | `sorrybench_experiment_runner.py` (flags) | `flags.DEFINE_float("token_ches_max_cos", 2.0, ...)` | flag CLI |
| 11 | `sorrybench_experiment_runner.py:464` (`DPOConfig(...)`) | `token_ches_max_cos=FLAGS.token_ches_max_cos,` | passer le flag |
| 12 | `similarity_measures_experiment_runner.py:494` (`DPOConfig(...)`) + flag | mêmes lignes 10–11 | pour l'expérience UltraFeedback de la section 5 |

Total estimé : ≈ 12 lignes.

## Expérience de validation proposée

1. Section 5 (UltraFeedback, OLMo-1B) : prendre les 512 exemples au quantile 1 (les plus similaires, qui
   provoquent le plus de displacement), entraîner avec et sans filtre par token, et comparer
   `eval_metrics/train/preferred_logprobs_change_mean`. Si le filtre par token réduit la baisse de `log π(y⁺)`,
   l'idée est validée dans leur propre cadre.
2. Section 6 (SORRY-Bench) : DPO sur le dataset complet + filtre par token, comparé au filtrage par ln-CHES
   (19 exemples) et au dataset complet sans filtre, sur le taux de refus.

## Points à vérifier

- `output_hidden_states=True` renvoie les états de toutes les couches. Ils existent déjà pour le backward, mais
  sous FSDP/DeepSpeed il faut vérifier que la mémoire tient. Sinon, passer par un hook sur la norme finale.
- `get_prompt_mask` dépend de `assistant_token` / `prompt_end_positions_func` : `S_w` et `S_l` doivent couvrir
  exactement les tokens de réponse, comme dans `ches.py`.
- Le ref model n'a pas besoin des états cachés : le filtre est calculé avec le modèle entraîné (détaché),
  comme CHES est calculé avec le modèle courant.
