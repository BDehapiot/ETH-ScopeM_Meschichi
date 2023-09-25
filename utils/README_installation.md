This tutorial provides a step-by-step guide to install Python, coupled with Spyder, using the Conda environment manager.  

**Conda** environment manager allows to run Python code within a controlled environment that contains all necessary dependencies. It's a best practice to create a distinct environment for each Python-based projects. 

**Spyder** is an Integrated Development Environment (IDE), enabling users to easely execute and interact with Python codes. Although not mandatory, Spyder is highly recommended for beginners.  

Select your operating system:

<details> <summary>Windows</summary>  

### Download GitHub repository:  

1) Download this GitHub repository
([link](https://github.com/BDehapiot/{{ repo_name }}/archive/refs/heads/main.zip)) 

2) Unzip folder to a known location (e.g. `C:\Users\YourUsername\Desktop`)

### Install Miniforge:  

3) Download Miniforge installer for Windows
([link](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe))  

4) Run the downloaded `.exe` file and select the following options:    
    - *create start menu shortcuts*  
    - *add Miniforge3 to PATH environment variable*  

### Setup Conda environment: 

5) Run Miniforge Prompt from start menu shortcuts  

    The prompt should read:  
    `(base) C:\Users\YourUsername>`  
    `(base)` meaning that we are in our base environment  

6) Move to the downloaded GitHub repository using the `cd` command: 
    ```bash
    cd Desktop/{{ repo_name }}-main
    ```
    The prompt should change to reflect our current location:  
    `(base) C:\Users\YourUsername\Desktop\{{ repo_name }}-main>`

7) Create a new Conda environment (takes a few minutes): 
    ```bash
    mamba env create -f environment.yml
    ```

8) Activate the newly created environment:
    ```bash
    conda activate {{ env_name }}
    ```
    The prompt should now start with `({{ env_name }})`  
    `({{ env_name }}) C:\Users\YourUsername\Desktop\{{ repo_name }}-main>`

### Start and setup Spyder IDE: 

9) Start Spyder using the following command:
    ```bash
    spyder
    ```

10) Create a new Spyder project
    - Click the `Projects` > `New Project...`
    - Choose `Existing directory`
    - Select the GitHub repository using the folder icon
    - Click the `Create` button  

    Projects can be re-opened later with: `Projects` > `Recent Projects...`

</details>  

<details> <summary>MacOS</summary>  

### Download GitHub repository: 

1) Download GitHub repository
([link](https://github.com/BDehapiot/{{ repo_name }}/archive/refs/heads/main.zip)) 

2) Unzip folder to a known location (e.g. `~/Desktop`)

### Install Miniforge:  

3) Download Miniforge installer for MacOS 
([legacy](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh))
([M-Series](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)) 

4) Open terminal by typing `terminal` in Launchpad or Spotlight search  
    
    The prompt should read:  
    `YourUsername@MacBook-Pro ~ %`

5) Move to the downloaded Miniforge script using the `cd` command:  
    It is most likely located in the `Downloads` folder    
    ```bash
    cd ~/Downloads
    ```  
    
6) Run the following command to install Miniforge:  

    *Legacy*
    ```bash
    bash Miniforge3-MacOSX-x86_64.sh
    ```  
    
    *M-Series*
    ```bash
    bash Miniforge3-MacOSX-arm64.sh
    ```
    Follow the Terminal prompts to complete installation and accept default options  

### Setup Conda environment: 

7) Close and re-open your terminal  

    The prompt should now read:  
    `(base) YourUsername@MacBook-Pro ~ %`  
    `(base)` meaning that we are in our base environment  

8) Move to the downloaded GitHub repository: 
    ```bash
    cd Desktop/{{ repo_name }}-main
    ```
    The prompt should change to reflect our current location:  
    `(base) YourUsername@MacBook-Pro Desktop/{{ repo_name }}-main %`  

9) Create a new Conda environment (takes a few minutes):  
    ```bash
    mamba env create -f environment.yml
    ```

10) Activate the newly created environment:
    ```bash
    conda activate {{ env_name }}
    ```

    The prompt should now start with `({{ env_name }})`  
    `({{ env_name }}) YourUsername@MacBook-Pro Desktop/{{ repo_name }}-main %`  

### Start and setup Spyder IDE: 

11) Start Spyder using the following command:
    ```bash
    spyder
    ```

12) Create a new Spyder project
    - Click the `Projects` > `New Project...`
    - Choose `Existing directory`
    - Select the GitHub repository using the folder icon
    - Click the `Create` button  

    Projects can be re-opened later with: `Projects` > `Recent Projects...`


</details>  

<details> <summary>Linux</summary>  

</details>  
