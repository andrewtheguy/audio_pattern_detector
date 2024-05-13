#!/bin/bash
TAG=$(date -u '+%Y%m%d%H%M%S')-$(openssl rand -base64 4 | tr -d "=+/") docker buildx bake -f docker-bake.hcl image-all --push
