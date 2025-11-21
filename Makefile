install: permissions
	./install.sh

cluster: permissions
	./cluster-install.sh

run: permissions
	./run.sh

permissions:
	chmod +x install.sh
	chmod +x cluster-install.sh
	chmod +x run.sh
