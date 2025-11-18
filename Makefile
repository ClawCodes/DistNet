install: permissions
	./install.sh

cluster: permissions
	./cluster-install.sh

permissions:
	chmod +x install.sh
	chmod +x cluster-install.sh
