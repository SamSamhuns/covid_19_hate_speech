# Data

The labeled data was acquired from [Davidson et al](https://github.com/t-davidson/hate-speech-and-offensive-language). The associated paper can be found at [Automated Hate Speech Detection and the Problem of Offensive Language](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665)

Each data file contains 5 columns:

`count` = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

`hate_speech` = number of CF users who judged the tweet to be hate speech.

`offensive_language` = number of CF users who judged the tweet to be offensive.

`neither` = number of CF users who judged the tweet to be neither offensive nor non-offensive.

`class` = class label for majority of CF users.
  0 - hate speech
  1 - offensive  language
  2 - neither
