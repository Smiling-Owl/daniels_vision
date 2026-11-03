# Shot dataset data dictionary

This dictionary describes every column in the three Phase 4 CSV outputs. The build reads pinned StatsBomb Open Data event JSON for La Liga 2015/2016 (competition 11, season 27) at repository revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`. See the saved upstream README and licence under `data/raw/statsbomb/<commit>/` before redistributing source data or derived outputs.

The upstream repository is [StatsBomb Open Data](https://github.com/hudl/open-data). Coordinate interpretation follows the [StatsBomb Open Data specification, Appendix 2, page 34](https://raw.githubusercontent.com/hudl/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/doc/StatsBomb%20Open%20Data%20Specification%20v1.1.pdf): the pitch frame is 120 by 80, the attacked goal center is `(120, 40)`, and its posts are `(120, 36)` and `(120, 44)`. Coordinates are kept in their native units; distances are not metres. The source records coordinates relative to the attacking direction, so the builder does not flip home/away coordinates.

## `data/processed/shots.csv`

Each row is one shot event in periods 1 or 2. The goal label uses the exact recorded outcome; all other recognized shot outcomes become 0. `is_goal` and `shot_outcome` are label/audit fields. `statsbomb_xg` is a reference probability supplied by the provider; it is not ground truth and is excluded from model inputs. Only the nine columns listed in `docs/model_columns.json` are initial candidate features. Match identifiers support future match-grouped evaluation, not prediction.

| Column | Type | Source / derivation | Meaning and role | Missing-value policy |
|---|---|---|---|---|
| `match_id` | int64 | match.id | Group/audit only; never a model feature. | Required; reject missing. |
| `event_id` | string | event.id | Unique shot-event identifier. | Required; reject missing or duplicate. |
| `event_index` | int64 | position in match event list | Zero-based source-event order used for audit. | Required. |
| `match_date` | date | match.match_date | Match date as YYYY-MM-DD. | Required. |
| `home_team_id` | int64 | match.home_team.home_team_id | Home-team identifier. | Required. |
| `home_team_name` | string | match.home_team.home_team_name | Home-team name. | Required. |
| `away_team_id` | int64 | match.away_team.away_team_id | Away-team identifier. | Required. |
| `away_team_name` | string | match.away_team.away_team_name | Away-team name. | Required. |
| `team_id` | int64 | event.team.id | Team credited with the shot. | Required. |
| `team_name` | string | event.team.name | Team credited with the shot. | Required. |
| `opponent_id` | int64 | match home/away metadata | Opponent of the shooting team. | Required. |
| `opponent_name` | string | match home/away metadata | Opponent of the shooting team. | Required. |
| `player_id` | int64 | event.player.id | Player associated with the shot. | Required. |
| `player_name` | string | event.player.name | Player associated with the shot. | Required. |
| `period` | int64 | event.period | Only periods 1 and 2 are included. | Required; other periods counted as exclusions. |
| `minute` | int64 | event.minute | Event minute, including stoppage-time values. | Required. |
| `second` | int64 | event.second | Second within the event minute. | Required. |
| `possession_id` | int64 | event.possession | Possession identifier for audit/group context. | Required. |
| `is_home` | int64 (0/1) | comparison of event.team.id and match.home_team.home_team_id | 1 when the shooting team is the home side, otherwise 0. | Required. |
| `is_penalty` | int64 (0/1) | event.shot.type.name | 1 when shot type is Penalty, otherwise 0. | Required; false otherwise. |
| `x` | float64 | event.location[0] | Original StatsBomb horizontal coordinate; audit only. | Required; non-finite or absent coordinates fail. |
| `y` | float64 | event.location[1] | Original StatsBomb vertical coordinate; audit only. | Required; non-finite or absent coordinates fail. |
| `distance_sb_units` | float64 | derived from x,y and goal center (120,40) | Euclidean distance to the center of the attacked goal in native StatsBomb units; not metres. | Required and finite; invalid geometry fails. |
| `angle_sb_radians` | float64 | derived from x,y and goalposts (120,36),(120,44) | Unsigned angle between the goalpost vectors, in radians. | Required and finite; zero-length post vector fails. |
| `body_part` | string | event.shot.body_part.name | Body part used for the shot. | Required; reject missing. |
| `shot_type` | string | event.shot.type.name | StatsBomb shot type, including penalty when applicable. | Required; reject missing. |
| `technique` | string | event.shot.technique.name | StatsBomb shot technique. | Required; reject missing. |
| `play_pattern` | string | event.play_pattern.name | StatsBomb play-pattern label for the event. | Required; reject missing. |
| `under_pressure` | int64 (0/1) | event.under_pressure | Provider pressure indicator; 1 when flagged, 0 otherwise. Zero can mean NOT FLAGGED, not certain real-world absence. | Missing source flag is represented as 0 / NOT FLAGGED. |
| `first_time` | int64 (0/1) | event.shot.first_time | Provider first-time-shot indicator; 1 when flagged, 0 otherwise. Zero means NOT FLAGGED. | Missing source flag is represented as 0 / NOT FLAGGED. |
| `open_goal` | int64 (0/1) | event.shot.open_goal | Provider open-goal indicator; 1 when flagged, 0 otherwise. Zero means NOT FLAGGED. | Missing source flag is represented as 0 / NOT FLAGGED. |
| `off_camera` | int64 (0/1) | event.off_camera | Audit-only provider indicator that the shot was off camera; excluded from model inputs. | Missing source flag is represented as 0 / NOT FLAGGED. |
| `is_goal` | int64 (0/1) | event.shot.outcome.name | Binary supervised target: 1 only for Goal, 0 for other recognized outcomes. | Required; unknown or missing outcome fails. |
| `shot_outcome` | string | event.shot.outcome.name | Original outcome retained for label audit. | Required; recognized outcome only. |
| `statsbomb_xg` | float64 | event.shot.statsbomb_xg | StatsBomb xG retained as an external probability reference, never a model feature. | Required and finite; expected in [0,1]. |

## `data/processed/goal_reconciliation.csv`

One row per indexed match. Shot goals plus Own Goal Against credited to each beneficiary are compared with the official score. `own_goals_conceded_*` counts own goals conceded by that side; `own_goals_credited_*` counts the opposing side's Own Goal Against records credited to it. Own Goal For events are validated as corroboration and never added to scores.

| Column | Type | Source / derivation | Meaning and role | Missing-value policy |
|---|---|---|---|---|
| `match_id` | int64 | match.match_id | Match identifier. | Required. |
| `match_date` | date | match.match_date | Match date. | Required. |
| `home_team_id` | int64 | match.home_team.home_team_id | Home-team identifier. | Required. |
| `home_team_name` | string | match.home_team.home_team_name | Home-team name. | Required. |
| `away_team_id` | int64 | match.away_team.away_team_id | Away-team identifier. | Required. |
| `away_team_name` | string | match.away_team.away_team_name | Away-team name. | Required. |
| `official_home_goals` | int64 | match.home_score | Official home score. | Required. |
| `official_away_goals` | int64 | match.away_score | Official away score. | Required. |
| `shot_goals_home` | int64 | derived from eligible shot labels | Ordinary Goal-outcome shots scored by home side. | Zero when none. |
| `shot_goals_away` | int64 | derived from eligible shot labels | Ordinary Goal-outcome shots scored by away side. | Zero when none. |
| `own_goals_conceded_home` | int64 | Own Goal Against events where home team is the offending side | Count of own goals conceded by the home team; credited to away. | Zero when none. |
| `own_goals_conceded_away` | int64 | Own Goal Against events where away team is the offending side | Count of own goals conceded by the away team; credited to home. | Zero when none. |
| `own_goals_credited_home` | int64 | Own Goal Against events credited to home; cross-checked against Own Goal For | Count of own goals credited to the home team; used in score reconstruction. | Zero when none; Own Goal For event is corroboration and is not counted again. |
| `own_goals_credited_away` | int64 | Own Goal Against events credited to away; cross-checked against Own Goal For | Count of own goals credited to the away team; used in score reconstruction. | Zero when none; Own Goal For event is corroboration and is not counted again. |
| `reconciled_home_goals` | int64 | shot_goals_home + own_goals_credited_home | Reconstructed home score using one own-goal representation. | Required. |
| `reconciled_away_goals` | int64 | shot_goals_away + own_goals_credited_away | Reconstructed away score using one own-goal representation. | Required. |
| `reconciled` | boolean | comparison with official score | True only if both reconstructed scores equal official scores. | Required; build fails if false. |

## `data/processed/own_goals.csv`

One row per Own Goal Against record, which identifies the side benefiting from the own goal. A related Own Goal For event identifier is included where the source links one. This file is an audit trail; own goals are not included as ordinary shot rows or model labels.

| Column | Type | Source / derivation | Meaning and role | Missing-value policy |
|---|---|---|---|---|
| `match_id` | int64 | match.match_id | Match identifier. | Required. |
| `event_id` | string | Own Goal Against event.id | Unique event recording the goal against the offending team. | Required. |
| `event_index` | int64 | position in match event list | Zero-based source-event order. | Required. |
| `match_date` | date | match.match_date | Match date. | Required. |
| `period` | int64 | event.period | Match period. | Required. |
| `minute` | int64 | event.minute | Event minute. | Required. |
| `second` | int64 | event.second | Second within event minute. | Required. |
| `offending_team_id` | int64 | event.team.id | Team against which the own goal was recorded. | Required. |
| `offending_team_name` | string | event.team.name | Name of the team against which the own goal was recorded. | Required. |
| `beneficiary_team_id` | int64 | opponent from match home/away metadata | Opponent credited with the goal. | Required. |
| `beneficiary_team_name` | string | opponent from match home/away metadata | Name of the opponent credited with the goal. | Required. |
| `player_id` | int64 | event.player.id | Player associated with the own-goal record. | Nullable if source omits the player. |
| `player_name` | string | event.player.name | Player associated with the own-goal record. | Nullable if source omits the player. |
| `own_goal_for_event_id` | string | related Own Goal For event, if linked | Corroborating paired event identifier; never counted as an additional goal. | Nullable when no related event is present. |

## Missing values and flag interpretation

Missing required labels, coordinates, player and core shot features, unrecognized outcomes, malformed events, non-finite numeric values, and degenerate geometry stop the build. The builder does not silently delete a shot or impute numeric values. Three provider context flags are the exception: `under_pressure`, `first_time`, and `open_goal` can be absent because the source records the positive flag. The CSV uses `False` for an absent flag and documents that as **NOT FLAGGED**, not proof the real-world condition was absent.

The specification cautions that the off-camera annotation may be less accurate than other shot data; the column remains an audit field and is not a predictor. The script excludes non-regulation periods from the shot table and records their count in its quality report. It also reports the raw true/false/absent counts for `off_camera`, the absent counts for the three context flags, and how many shot events lack a freeze frame. These are audit facts only; no rows are dropped because of missing freeze frames. It retains the source JSON unchanged and reports source/output hashes in `docs/shot_dataset_report.json`.
