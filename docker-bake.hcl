variable "TAG" {
}

variable "IMAGE_NAME" {
    default="ghcr.io/andrewtheguy/audio_pattern_detector"
}


group "default" {
    targets = ["worker"]
}

target "worker" {
    dockerfile = "Dockerfile"
    # both the tag and latest
    tags = ["${IMAGE_NAME}:${TAG}","${IMAGE_NAME}"]
}

target "worker_multi" {
    inherits = ["worker"]
    platforms = ["linux/amd64", "linux/arm64"]
}

target "worker_amd64_only" {
    inherits = ["worker"]
    platforms = ["linux/amd64"]
}

group "image-all" {
    targets = ["worker_multi"]
}