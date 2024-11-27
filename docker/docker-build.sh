source .env

cp ../requirements.txt .

docker compose -p $COMPOSE_PROJECT_NAME -f docker-compose.yml build
