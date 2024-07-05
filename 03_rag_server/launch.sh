#!/bin/bash
##########################################
# ENV VARIABLES 
# USER EDIT BLOCK
export MODELPATH="INSERT_PATH"  # Path to LLM
export PDFDIRPATH="INSERT_PATH"  # Path to directory storing PDFs
export EMBEDMODELPATH="INSERT_PATH" # Path to embedding models
##########################################

# Set tensor parallel size
export TENSORPAR=1

# Set path to the Langchain app
export APPPATH="./llsc-rag"

# Set visible devices
export CUDA_VISIBLE_DEVICES="0"

# Get address and port of head node
# export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_HOST=$(hostname -s)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export HEAD_NODE_ADDR="$MASTER_HOST:$MASTER_PORT"

echo "MASTER ADDR: $MASTER_HOST"
echo "MASTER PORT: $MASTER_PORT"

# Set HF to use offline mode
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Create TMPDIR
export TMPDIR=/state/partition1/user/$USER
export RAY_TMPDIR=$TMPDIR/raytemp
mkdir -p $RAY_TMPDIR

# Start head server
echo "STARTING RAY HEAD SERVER"
ray start -v --head --block \
     --dashboard-host 0.0.0.0 \
     --port=$MASTER_PORT \
     --num-cpus 5 \
     --num-gpus 1 \
     --temp-dir=$RAY_TMPDIR &

sleep 20

# Check status
ray status

# Setup forwarding for Langchain
#export LC_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export LC_PORT=8000
export PORTAL_FWNAME="$(id -un | tr '[A-Z]' '[a-z]')-langchain"
export PORTAL_FWFILE="/home/gridsan/portal-url-fw/${PORTAL_FWNAME}"
echo $PORTAL_FWFILE
echo "Portal URL is: https://${PORTAL_FWNAME}.fn.txe1-portal.mit.edu/playground/"
echo "http://$MASTER_HOST:$LC_PORT" > $PORTAL_FWFILE
cat $PORTAL_FWFILE
chmod u+x ${PORTAL_FWFILE}

# Launch Langchain
cd $APPPATH
langchain serve --host 0.0.0.0 --port $LC_PORT &

# Run server
echo "START vLLM SERVER"
python -m vllm.entrypoints.api_server \
     --model $MODELPATH \
     --tensor-parallel-size $TENSORPAR \
     --port 8080 \
     --enforce-eager

echo "ALL SERVERS STARTED"
