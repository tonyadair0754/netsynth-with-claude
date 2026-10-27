# Put the Social Capital Atlas file here (not committed)

Save the Atlas "friending rates" CSV as `100_x_100_friendship_matrix.csv` in this folder (or pass any path with `--atlas PATH`).
Get it from <https://data.humdata.org/dataset/social-capital-atlas> (the file is called `friending_rates.csv` there).

Expected columns: `own_ses_percentile` plus `friend_prob_p1` ... `friend_prob_p100` (100 rows, one per own-SES percentile).
Check the dataset's license/terms on that page before redistributing it; that is why it is not part of this repository.
