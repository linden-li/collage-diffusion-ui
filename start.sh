if [[ "$1" == "frontend" ]]; then
  echo "Starting frontend"
  cd frontend 
  rm -f nohup.out
  npm run build
  nohup serve -s build -l 3000 &
fi 

# Assumes that we have already `conda activate diffusion`
if [[ "$1" == "backend" ]]; then
  echo "Starting backend"
  source ./env/bin/activate
  cd backend
  rm -f nohup.out
  nohup python3 api.py &
fi 
