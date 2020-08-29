#!/bin/sh

for i in $(seq 1 10)  
do   
. ./installDL.sh 
done
 
cp -r analysePro/averageResult averageResult
chmod u+x averageResult
./averageResult DLFinalresult.txt
rm -f averageResult
rm -f DLFinalresult.txt

for i in $(seq 1 10)  
do   
. ./installDL_resample.sh 
done
 
cp -r analysePro/averageResultResample averageResultResample
chmod u+x averageResultResample
./averageResultResample DLFinalresult.txt
rm -f averageResultResample
rm -f DLFinalresult.txt
