#!/bin/bash

#replace with your roll no
ROLL_NO="cs24resch11013"
#parent dir is the location of your submission folder eg. if my submission folder is /home/srinivas/FoML/cs21resch01004_foml24_hackathon
PARENT_DIR="/home/darpan/Desktop/7thSem/AI2000/FoML"
SUBMISSION_DIR="${PARENT_DIR}/${ROLL_NO}_foml24_hackathon"
#location of predictions generated
OUTPUT_DIR="${PARENT_DIR}/outputs"
#location for trianing traces
LOGS_DIR="${PARENT_DIR}/logs"

#location of the test file
PRIVATE_TEST='/home/darpan/Desktop/7thSem/AI2000/FOML_hkthn/test.csv'
#location fo the train file
TRAIN_PATH="/home/darpan/Desktop/7thSem/AI2000/FOML_hkthn/train.csv"

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

#cd into parent dir
cd "${SUBMISSION_DIR}"

#make output dir if doesnt exist
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

#copy over train.csv if doesn't exist
if [ ! -f "${SUBMISSION_DIR}/train.csv" ]; then
	cp "${TRAIN_PATH}" .
	echo "train file copied"
else
	echo "train file present"
fi

py_file=$(find "." -type f -iname "*.py" | head -n 1)
py_filename=$(basename "${py_file}")
python "${py_filename}" --test-file "${PRIVATE_TEST}" --predictions-file "${OUTPUT_DIR}/${ROLL_NO}_private_test_predictions.csv" > "${LOGS_DIR}/${ROLL_NO}_log.txt"