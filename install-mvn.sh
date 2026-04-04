module load openjdk/21

# Download Maven 3.9.13
wget https://archive.apache.org/dist/maven/maven-3/3.9.13/binaries/apache-maven-3.9.13-bin.tar.gz

# Extract it
tar -xzf apache-maven-3.9.13-bin.tar.gz

# Add it to your PATH so your Python script can find 'mvn'
export PATH=$PWD/apache-maven-3.9.13/bin:$PATH
