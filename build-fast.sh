#!/bin/bash

# Fast build script with optimizations
set -e

VERSION="${1:-1.0.3}"
ARCHITECTURES=("amd64" "aarch64")

echo "🚀 Starting fast parallel builds for version $VERSION"

# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Function to build single architecture
build_single() {
    local arch=$1
    local variant=$2
    local dir=$3
    
    echo "🔨 Building $variant for $arch..."
    
    # Use build cache and parallel processing
    podman build \
        --build-arg BUILD_FROM=python:3.11-bookworm \
        --build-arg BUILD_ARCH=$arch \
        --build-arg BUILD_VERSION=$VERSION \
        --tag dao-$variant-$arch:$VERSION \
        --tag dao-$variant-$arch:latest \
        --cache-from dao-$variant-$arch:latest \
        --jobs 4 \
        $dir/
    
    echo "✅ $variant ($arch) build completed"
}

# Build both variants in parallel using background processes
echo "🏗️  Building DAO Modern..."
build_single "amd64" "modern" "dao-modern" &
MODERN_PID=$!

echo "🪶 Building DAO Light..."  
build_single "amd64" "light" "dao-light" &
LIGHT_PID=$!

# Wait for both builds to complete
wait $MODERN_PID
MODERN_RESULT=$?

wait $LIGHT_PID  
LIGHT_RESULT=$?

# Check results
if [ $MODERN_RESULT -eq 0 ] && [ $LIGHT_RESULT -eq 0 ]; then
    echo "🎉 All builds completed successfully!"
    
    # Show built images
    echo "📦 Built images:"
    podman images | grep "dao-.*:$VERSION"
    
    exit 0
else
    echo "❌ Some builds failed:"
    [ $MODERN_RESULT -ne 0 ] && echo "  - DAO Modern failed"
    [ $LIGHT_RESULT -ne 0 ] && echo "  - DAO Light failed"
    exit 1
fi