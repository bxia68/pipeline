FROM postgres
# COPY macrostrat_dump.sql /docker-entrypoint-initdb.d

ENV POSTGRES_DB=output_db
ENV POSTGRES_USER=admin
ENV POSTGRES_PASSWORD=admin