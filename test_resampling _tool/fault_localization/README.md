#A project of test cases' resampling to enhance fault localization.
# Usage
1. Requirements
python >= 3.6.2
tensorflow >=1.3.0
gcc: 5.4.0

2. Run project
test_resampling: python resampling.py dev
fault_localization :./runall_NN.sh

3. Result
test_resampling: covMatrix_resample.txt(statements' coverage information matrix after resampling)
 	         error_resample.txt(test cases' results after resampling)
fault_localization :faultStatementRank.txt(fault localization result with ranks)
