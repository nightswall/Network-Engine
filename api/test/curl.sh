FILE="test30.csv"

# set the URL of the web server
#URL="http://172.18.0.4:8000/api/predict/temperature" #for docker
URL="http://localhost:8002/api/predict/network" #for local

# read the CSV file line by line
while read -r line; do
  # send the line to the web server using curl
  # echo "$line"
  curl -X POST -d "data=$line" "$URL"
  sleep 1
done < "$FILE"

