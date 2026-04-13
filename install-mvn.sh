#!/usr/bin/env bash

set -euo pipefail

MVN_VERSION="3.9.13"
MVN_DIR="apache-maven-${MVN_VERSION}"
MVN_TGZ="${MVN_DIR}-bin.tar.gz"
MVN_URL="https://archive.apache.org/dist/maven/maven-3/${MVN_VERSION}/binaries/${MVN_TGZ}"

if command -v module >/dev/null 2>&1; then
	module load openjdk/21 || true
fi

if [[ ! -x "${MVN_DIR}/bin/mvn" ]]; then
	echo "Installing Maven ${MVN_VERSION} in ${PWD}/${MVN_DIR}"
	if [[ ! -f "${MVN_TGZ}" ]]; then
		wget "${MVN_URL}"
	fi
	tar -xzf "${MVN_TGZ}"
else
	echo "Maven already present at ${PWD}/${MVN_DIR}/bin/mvn"
fi

export MAVEN_HOME="${PWD}/${MVN_DIR}"
export PATH="${MAVEN_HOME}/bin:${PATH}"

echo "MAVEN_HOME=${MAVEN_HOME}"
command -v mvn
mvn -v
