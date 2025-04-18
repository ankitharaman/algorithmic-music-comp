#!/usr/bin/env bash

if [[ $1 == "--build" ]] then
    docker build -t amc_final_project .
else
    docker run -it -v $(pwd):/app amc_final_project
fi
