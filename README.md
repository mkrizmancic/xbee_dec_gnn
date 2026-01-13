# Instructions

## Installation and setup

1. Prepare SSH keys for remote Raspberry Pi access:
   1. On your host machine (not inside the Docker container), generate SSH keys if you don't have them already:
      ```bash
      ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C <your_name_or_email>"
      ```
   1. Add the SSH key to the SSH agent:
      ```bash
      eval "$(ssh-agent -s)"
      ssh-add ~/.ssh/id_ed25519
      ```
   1. Copy the public key to each Raspberry Pi you want to connect to. This unfortunately requires a bit of manual work:
      ```bash
      ssh-copy-id -i ~/.ssh/id_ed25519.pub pi@rpi0.local
      ssh-copy-id -i ~/.ssh/id_ed25519.pub pi@rpi1.local
      ssh-copy-id -i ~/.ssh/id_ed25519.pub pi@...
      ```

1. Install Docker by following the instructions below.

   1. You must have Ubuntu OS installed on your computer. If you have an NVIDIA GPU, please follow [these instructions](https://github.com/larics/docker_files/wiki/2.-Installation#gpu-support) to prepare for Docker installation.
   1. Follow these [instructions](https://docs.docker.com/engine/install/ubuntu/) to install the Docker engine.
   1. Then follow these [optional steps](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) to manage docker as a non-root user. If you skip this, every `docker` command will have to be executed with `sudo`. Skip the _"Note: To run Docker without root privileges, see Run the Docker daemon as a non-root user (Rootless mode)."_ part. This is just a note and we do not need it.
   1. Enable running graphical applications in the container by executing:
        ```bash
        xhost +local:docker
        ```
   1. Make this option permanent by adding the command to your `.profile` file which executes on every login.
        ```bash
        echo "xhost +local:docker > /dev/null" >> ~/.profile
        ```

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/mkrizmancic/xbee_dec_gnn.git
   ```

1. Navigate to the project directory:

   ```bash
   cd xbee_dec_gnn
   ```

1. Build the Docker image:

   ```bash
   docker build -t xbee_gnn_img .
   ```

1. Run the Docker container for the first time:

   ```bash
   ./docker/run_docker.sh
    ```
   You will see your prompt change from `<your username>@<your hostname>` to `root@<your hostname>`. This indicates that you are now inside the Docker container.

1. Later, you can start the container again with:

   ```bash
   docker start -i xbee_gnn_cont
   ```

1. You can exit the container at any time by typing `exit` or pressing `Ctrl+D`.

## Running a demo
This code comes with a demo of decentralized GNN execution:
- Uses a pretrained model and a prepared dataset.
- A central node sends the graph structure and initial features to each node.
- Based on the input graph and features, the decentralized GNN model finds a solution to the MIDS problem.
- Each node outputs 1 if it thinks it is a part of the MIDS set, otherwise 0.
- Nodes communicate over ROS 2.
- It can be run locally on a single machine or on Raspberry Pi devices connected to the same network.

To run the demo, follow these steps:
1. Inside the Docker container, navigate to the ROS2 package and its launch directory:

   ```bash
   cd ~/ros2_ws/src/ros2_dec_gnn/ros2_dec_gnn/launch
   ```
1. Launch the demo with the following command:

   ```bash
   tmuxinator start -p tmux_launch_mids.yml <option>
   ```
   If you want to run the demo locally on your machine, replace `<option>` with `local`. If you want to run the demo on Raspberry Pis connected to the same network, replace `<option>` with `remote`.

1. You will see 5 tmux panes, each representing a node in the graph. Each pane will display logs of the node's operations, including the final output indicating whether the node is part of the MIDS set (1) or not (0). A graphical window displaying the graph and the MIDS solution(s) will also appear. You can change the current graph by clicking the "Next" button in the graphical window.

1. Stop the demo by doing the following:
   1. Stop GNN nodes with `Ctrl+C`. (You can skip this in local mode, but in remote mode, it shuts down LEDs on Raspberry Pis.)
   1. Kill the tmux session with `Ctrl+A` followed by `K` in the terminal.

## Developing the Xbee package
1. Use the ROS 2 demo as the inspiration for Xbee variant. The code should be almost identical, except for the communication part which should use Xbee instead of ROS 2.

1. During Docker container creation, the `xbee_dec_gnn` package is **copied** into the container. This means that any change you make in this directory on your machine is going to be different than what is inside the container. The package is copied to the `~/other_ws/xbee_dec_gnn` directory.
   1. Attach the VS Code to the Docker container and navigate to your project directory.
   1. Make necessary changes.
   1. When you want to save your changes, simply commit and push them to GitHub from the Docker terminal or the VS Code. The Docker is set up so that it uses your SSH agent from the host.
   1. To run the code, navigate to the appropriate directory and run it in the terminal inside the Docker container.
   1. You will need to create or modify the tmuxinator configuration files in the `launch` directory to run your code in multiple panes.

1. Datasets and models are **mounted** into the container from the directories inside `docker/volumes` on your host machine to the `/root/resources/data` and `/root/resources/models` directories inside the container. This means that any changes made on the host are immediately visible inside the container and vice versa. Your code should use these locations to load data and models. To change or add new datasets or models, simply add them to the respective directories on your host machine.

1. When you are happy with your changes and want to test them on Raspberry Pis, push the code to GitHub from the container on your host machine and pull them inside the container on the Raspberry Pis.
   1. Use the tmuxinator config inside the launch directory to SSH into all Raspberry Pis in parallel:
      ```bash
      cd ~/other_ws/xbee_dec_gnn/launch
      tmuxinator start -p tmux_ssh_pi.yml
      ```
   1. You are now connected to each Raspberry's terminal and they are all synchronized (i.e., commands typed in one terminal are replicated in all others). Start the docker container on each Raspberry Pi:
      ```bash
      docker start -i xbee_gnn_cont
      ```
   1. Navigate to the `xbee_dec_gnn` package and pull the latest changes:
      ```bash
      cd ~/other_ws/xbee_dec_gnn
      git pull
      ```
  1. Exit the container by pressing `Ctrl+D` and then exit the SSH sessions by pressing `Ctrl+D` again.

1. Run the Xbee GNN code remotely on all Raspberry Pis using the tmuxinator config:
   ```bash
   cd ~/other_ws/xbee_dec_gnn/launch
   tmuxinator start -p tmux_launch_mids.yml remote
   ```
   This command automatically SSHs into all Raspberry Pis in parallel, starts the Docker container, and runs the Xbee GNN code on each device.

1. To upload new or modified datasets and models to the Raspberry Pis, use the provided bash script:
   ```bash
   ./docker/volumes/copy_to_rpi.sh
   ```

## Summary
1. Modify the Xbee code to match the functionality of the ROS 2 MIDS demo. Replace ROS 2 communication with Xbee communication. All changes should be made inside the Docker container on your host machine.
1. When ready, push changes to GitHub from the Docker container on your host machine. SSH into all Raspberry Pis in parallel using tmuxinator, start the Docker container on each Pi, and pull the latest changes from GitHub.
1. Run the Xbee GNN code on all Raspberry Pis using tmuxinator.
1. Repeat steps 1-3 as necessary.
1. Develop your own datasets and models, and add them to the `docker/volumes` directory on your host machine.
1. Create a new branch in `ros2_dec_gnn` for your application and modify `node.py` code as necessary. Test the decentralized GNN on your dataset and model using ROS 2 inside the Docker container on your host machine.
1. Test the decentralized GNN on your dataset and model using Xbee on Raspberry Pis. Follow the previous instructions to copy the dataset and models.

## Phases/tasks
- [ ] Try running the provided ROS 2 MIDS demo locally inside the Docker container on your host machine.
- [ ] Try running the provided ROS 2 MIDS demo remotely on Raspberry Pis.
- [ ] Modify the ROS 2 MIDS demo code to use Xbee communication instead of ROS 2.
- [ ] Create your own datasets and models for the decentralized GNN.
- [ ] Test your application using ROS 2 locally and on Raspberry Pis.
- [ ] Test your application using Xbee on Raspberry Pis.

## Bonus section
The provided Docker image comes with a few preinstalled tools and configs which may simplify your life.

**Tmuxinator** is a tool that allows you to start a tmux session with a complex layout and automatically run commands by configuring a simple yaml configuration file. Tmux is a terminal multiplexer - it can run multiple terminal windows inside a single window. This approach is simpler than having to do `docker exec` every time you need a new terminal.

You don't need to write new configuration files for your projects, but some examples will use Tmuxinator. You can move between terminal panes by holding down `Ctrl` key and navigating with arrow keys. Switching between tabs is done with `Shift` and the arrow keys. If you have a lot of open panes and tabs in your tmux, you can simply kill everything and exit by pressing `Ctrl+A` and then `K`.

Here are some links: [Tmuxinator](https://github.com/tmuxinator/tmuxinator), [Getting starded with Tmux](https://linuxize.com/post/getting-started-with-tmux/), [Tmux Cheat Sheet](https://tmuxcheatsheet.com/)

**Ranger** is a command-line file browser for Linux. While inside the Docker container, you can run the default file browser `nautilus` with a graphical interface, but it is often easier and quicker to view the files directly in the terminal window. You can start ranger with the command `ra`. Moving up and down the folders is done with the arrow keys and you can exit with a `q`. When you exit, the working directory in your terminal will be set to the last directory you opened while in Ranger.

**Htop** is a better version of `top` - command line interface task manager. Start it with the command `htop` and exit with `q`.

**VS Code** - If you normally use VS Code as your IDE, you can install [Dev Containers](https://code.visualstudio.com/docs/remote/containers#_sharing-git-credentials-with-your-container) extension, which will allow you to continue using it inside the container. Simply start the container in your terminal (`docker start -i mrs_project`) and then attach to it from VS Code (open action tray with `Ctrl+Shift+P` and select `Dev Containers: Attach to Running Container`).