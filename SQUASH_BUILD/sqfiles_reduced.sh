#!/bin/bash
DS=${1}
DS_BASE=${2}
echo ${1}
echo ${2}
PROOT=partitions
AROOT=allocators
echo ${1}/${PROOT}
echo ${1}/${AROOT}

pfiles=$(find ${DS}/${PROOT} -type f -name "${DS_BASE}" -o -name "${DS_BASE}.af")

allfiles="$rfiles $pfiles $afiles"
echo $allfiles | sed 's/ /\n/g' > filelist.txt

zipfile=${DS}_reduced_reqfiles.zip
zip $zipfile -@ < filelist.txt