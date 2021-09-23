# Twint Setup/Initial Configuration Guide
## Spyder IDE Config:
First we must clone the twint GitHub repo and install the necessary files. Run the following commands in your Python Console:
```sh
!git clone --depth=1 https://github.com/twintproject/twint.git
cd twint
!pip3 install . -r requirements.txt
```
To fix twint functionality we must upgrade the our version by running:
```sh
!pip3 install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint
```
Now we have a working version of twint. To access the proper libraries we need to import the following items in our Python program if we want to use twint and apply the 'nest_asyncio' library:
```py
import twint
import nest_asyncio

nest_asyncio.apply()
```
Now you are set to begin using Twint. Below is a simple example of a web-scrape of 20 most recent tweets from the @nbcnews account:
```py
c = twint.Config()
c.Search = "from:@nbcnews"
c.Limit = 20
twint.run.Search(c)
```
