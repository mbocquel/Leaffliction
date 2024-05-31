SOURCES =	docker-compose.yml \

all:     $(SOURCES)
		docker compose -f ./docker-compose.yml up


down:    $(SOURCES)
		docker compose -f ./docker-compose.yml down

clean:    $(SOURCES)
		docker compose -f ./docker-compose.yml down -v

fclean:    clean
		docker system prune -af

test:
		make test -C backend_fastAPI

re:        fclean all
.PHONY: all re down clean