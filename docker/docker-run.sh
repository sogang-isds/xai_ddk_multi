source .env

SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# check environment settings
if [ "$ENVIRONMENT" = "prod" ]; then
  bash $SCRIPT_DIR/check-env.sh || exit 1
fi

docker compose -p $COMPOSE_PROJECT_NAME up -d
