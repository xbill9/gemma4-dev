#!/bin/bash
# llama-server with one shared KV cache across slots (-kvu); everything else from lib.sh serve().
exec /home/xbill/llama.cpp/build/bin/llama-server "$@" -kvu
