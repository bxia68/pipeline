FROM postgres
# COPY macrostrat_dump.sql /docker-entrypoint-initdb.d

ENV POSTGRES_USER=admin
ENV POSTGRES_PASSWORD=admin
ENV POSTGRES_DB=output_db