# CS221_Project  

**Project install**  

```bash
git clone https://github.com/PhilippeW83440/CS221_Project.git
cd CS221_Project
pip install -r requirements.txt
cd gym-act
pip install -e .
```
Then check you can run the notebook Test_Setup.ipynb without error

### Using a virtual env

We recommend using python3. You may find convenient to use a virtual env.

```bash
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with ```deactivate```.

## Codalab
TODO: use https://codalab.org/  
The important thing that CodaLab provides is full reproducibility thanks to docker - a certificate that your code runs on a particular dataset with particular version of libraries/code and generates the reported results.  
