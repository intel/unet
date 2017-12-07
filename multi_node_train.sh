# Copy train script and settings files
pscp.pssh -h ~/hosts.txt /home/bduser/mattson/GE/unet/train_dist.py /home/bduser/mattson/GE/unet/
pscp.pssh -h ~/hosts.txt /home/bduser/mattson/GE/unet/settings_dist.py /home/bduser/mattson/GE/unet/

# Read worker host array from hosts.txt
readarray a < ~/hosts.txt

# Iterate through hosts and execute appropriate scripts.
start = 0
let "worker = start"
for i in "${a[@]}"
do
	echo "$i"
	echo "$worker"
	let "worker += 1"
done
