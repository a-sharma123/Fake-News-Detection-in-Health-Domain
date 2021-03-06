# FakeHealth
This repository (FakeHealth) is collected to address challenges in Fake Health News detection, which includes news contents, news reviews, social engagements and user network.

## Overviews
Our repository consist of two datasets: HealthStory and HealthRelease. Due to the twitter policy of protecting user privacy, the fullcontents of user social engagements and network are not al-lowed to directly publish. Instead, we store the IDs of all so-cial engagements into json files, and supplement them witha API to trivially attain the social engagements and user net-work from twitter The IDs are stored in ./dataset/engagements/HealthRelease.json and ./dataset/engagements/HealthStory.json.  

## Requirements
* twython==3.7.0
* [Developer APP](https://developer.twitter.com/en/docs/basics/apps/overview) of twitter to generate app_key,app_secret,oauth_token and oauth_token_secret

## Running Code
1. set the .\API\resources\tweet_keys_file.txt in the format of: <br> 
app_key,app_secret,oauth_token,oauth_token_secret<br>
XXXXXX,XXXXXXX,XXXXXXXXX,XXXXXXXXXXXXX
2. Build HealthStory:<br>
   <p> python main.py news_type=HealthStory sav_dir=../dataset <p>
3. Build HealthRelease:<br>
   <p> python main.py news_type=HealthRelease sav_dir=../dataset <p>
## Data Format
The data provided here only cantain the 
The downloaded dataset will have the following  folder structure,
* content
  * HealthStory
    * \<news_id>.json: a list of news contents wich include **URL**, **Title**, **Key words**, **Tags**, **Image URL**, **Author** and **Publishing Date**.
  * HealthRelease.json: ~
* reviews
  * HealthStory.json: a list of news reviews which include **Rating**, **news source**,**description**, **summary of the review**, **ground truth labels of the ten standard criteria**, **explanations of the criteria judgements** and **image link**. 
  * HealthRelease.json: ~
* engagements
  * HealthStory
    * \<news_id>
      * tweets
        * \<ID>.json: The json file of the tweet object. The detailed attributes of tweet object is [here](https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object).
        * ......
      * retweets
        * \<ID>.json
        * ......
      * replies
        * \<ID>.json
    * HealthRelase
      * ......
* user_network
  * user_profiles
    * \<user_name>.json: The json file of the user profile object. The detailed attributes of user profile object is [here](https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/user-object)
    * ......
  * user_timelines
    * \<user_name>.json: a list of tweet objects
    * ......
  * user_followers
    * \<user_name>.josn: a list of user followers profiles
    * ......
  * user_following
    * \<user_name>.json: a list of user following profiles
    * ......



   
   

