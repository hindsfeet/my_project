/* 20221104 - Initial Base Version by Minnie Tan */
Check documentation:
1. Read Batch Machine Learning.docx  
2. To use the software tool: Check Software use
   option 1: pipenv install 
   - this will use the pipfile.lock file
   option 2: install the modules
   pip install pandas numpy yellowbrick nltk sklearn varname apyori tensorflow seaborn matplotlib sklearn_som torch category_encoders openpyxl
3. After go to the root project folder <my_project> with the global configuration in config.json and all the input template files in the specified input folder, the default setup is <my_project/input>
   template files: <my_project/input/input_*.txt>
   * Change the template files according, one record for one input file processiong, but at times, code modification is needed
   * y - is the last column usually
   regression - <input_regression.txt>
   classification - <input_classifier.txt>
   cluster - <input_cluster.txt>
   rules - <input_rules.txt>
   sample - <input_sample.txt>
   text - <input_text.txt>
   neural_network - <input_neural_network.txt>
   unsupervised (for SOM)- <input_unsupervised.txt> 
   unsupervised1 (for RBM)- <input_unsupervised1.txt>
 
4. Execute the input files using python command
   python -W ignore main.py -i <select one input file>  
   
  
   