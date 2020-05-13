#!/bin/sh

export subject_dir=Math
export version=59
echo copying to coverage_info
cp -r ${subject_dir}/result/v${version}/componentinfo.txt Coverage_Info
cp -r ${subject_dir}/result/v${version}/covMatrix.txt Coverage_Info
cp -r ${subject_dir}/result/v${version}/error.txt Coverage_Info


echo resampling covMatrix and error

cd Coverage_Info
python resampling.py
cd ..

echo excuting python NN.py
python NN.py dev_resampling

cd Coverage_Info
echo excuting formatDLResult.c
./formatDLResult
cd ..

echo moving DL_result.txt to result
mv Coverage_Info/DL_result.txt ${subject_dir}/result/v${version}

cd ${subject_dir}
echo excuting DL.c
cp -r analysePro/DL DL
./DL result/v${version}/componentinfo.txt result/v${version}/DL_result.txt
rm -f DL

echo moving DeepLearning.txt to result
mv DeepLearning.txt result/v${version}/DL_result

echo excuting sliceDL.c
cp -r analysePro/sliceDL sliceDL
./sliceDL result/v${version}/componentinfo.txt result/v${version}/DL_result.txt sliceResult/v${version}/sliceResult.txt
rm -f sliceDL
echo moving SliceDeepLearning.txt to result
mv SliceDeepLearning.txt result/v${version}/DL_result
cd ..

echo get final result
cp -r analysePro/translate translate
./translate ${subject_dir}/result/v${version}/DL_result/DeepLearning.txt ${subject_dir}/result/v${version}/DL_result/SliceDeepLearning.txt
rm -f translate

rm -f ${subject_dir}/result/v${version}/DL_result.txt
rm -f ${subject_dir}/result/v${version}/DL_result/DeepLearning.txt
rm -f ${subject_dir}/result/v${version}/DL_result/SliceDeepLearning.txt
rm -f Coverage_Info/componentinfo.txt 
rm -f Coverage_Info/covMatrix.txt 
rm -f Coverage_Info/error.txt 
rm -f Coverage_Info/covMatrix_resample.txt 
rm -f Coverage_Info/error_resample.txt 
rm -f Coverage_Info/DL_result_unformat.txt 
