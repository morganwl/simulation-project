FROM python:3.10 as build
RUN pip install build
ADD pyproject.toml .
ADD src ./src
RUN python3 -m build

FROM python:3.10
ADD requirements.txt .
RUN pip install -r requirements.txt
COPY --from=build dist ./dist
RUN pip install dist/freebus-*.tar.gz
RUN python
