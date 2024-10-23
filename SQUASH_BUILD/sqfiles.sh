#!/bin/bash
DS=${1}
DS_BASE=${2}
echo ${1}
echo ${2}
PROOT=partitions
AROOT=allocators
echo ${1}/${PROOT}
echo ${1}/${AROOT}

rfiles=$(find ${DS} -type f -name "${DS_BASE}.afstdq" -o -name "${DS_BASE}.qavars.npz")
pfiles=$(find ${DS}/${PROOT} -type f -name "${DS_BASE}" -o -name "${DS_BASE}.af" -o -name "${DS_BASE}.tf" -o -name "${DS_BASE}.vaq" -o -name "${DS_BASE}.qpvars.npz")
afiles=$(find ${DS}/${AROOT} -type f -name "${DS_BASE}_qry.npz")

allfiles="$rfiles $pfiles $afiles"
echo $allfiles | sed 's/ /\n/g' > filelist.txt

zipfile=${DS}_reqfiles.zip
zip $zipfile -@ < filelist.txt