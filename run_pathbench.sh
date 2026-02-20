#Set slideflow backends
export SF_SLIDE_BACKEND=cucim
export SF_BACKEND=torch

#Set the config file
CONFIG_FILE=$1

#Run the program
python3 main.py --config $CONFIG_FILE