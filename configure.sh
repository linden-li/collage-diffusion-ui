
config_file="./configs/${1}"

echo "Attempting to apply config: ${config_file}"

if [ -f "${config_file}" ] 
then
    rm -f ./config.json && ln -s ${config_file} ./config.json && echo "Success."
else
    echo "Error: File ${config_file} does not exists."
fi
