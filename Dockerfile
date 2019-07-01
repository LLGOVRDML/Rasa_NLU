FROM rasa_baseimage
COPY . /usr/local/src/
RUN chmod +x /usr/local/src/entrypoint.sh
ENTRYPOINT ["/usr/local/src/entrypoint.sh"]