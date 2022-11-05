#!bin/bash 
#----------------------------------------------------#
# Download data from kaggle Dataset,
# one has to configure the API and save kaggle.json 
# under home directory to be able to use this script.
#----------------------------------------------------#
DATA_NAME="medical_appointment.csv"
DATA_ZIP="medical-appointment-kaggle.zip"
DATA_DIR="data"

if test -e ${DATA_ZIP}; then
    rm ${DATA_ZIP}
fi

kaggle datasets download -d pierricm/medical-appointment-kaggle

unzip -p ${DATA_ZIP} > ${DATA_NAME}

if [ -d ${DATA_DIR} ]; then
    echo "${DATA_DIR} already exists."
else
    mkdir ${DATA_DIR}
fi

echo "Copying ${DATA_NAME} into ${DATA_DIR} folder."
mv ${DATA_NAME} ${DATA_DIR}