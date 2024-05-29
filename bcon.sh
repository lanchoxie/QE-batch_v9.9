#!/bin/bash

# $1: output
# $2: input
# ./bcon.sh output input

zero=0
nfile=$(find . -maxdepth 1 -type f -name "$1.*" | wc -l)
nfile_tmp=$(find . -maxdepth 1 -type f -name "$2.*" | wc -l)

if [ $nfile -ne $nfile_tmp ]
then
   echo "Inconsistent input and output"
fi

if [ $nfile -ne $zero ]
then
   noutput=$( ls -l $1.* | wc -l)
   let c=$noutput+1
   cp $1 $1.${c}
   cp $2 $2.${c}
else
   cp $1 $1.1
   cp $2 $2.1
fi

natom=$(grep -e 'number of atoms/cell' $1 | tail -n 1 |awk '{print $5}')
na=$(grep -e 'ATOMIC_POSITIONS (' $1 | wc -l)
nc=$(grep -e 'CELL_PARAMETERS (' $1 | wc -l)
if [ $na -eq $zero ]
then
    echo "No relaxation found, Exit"
    exit
fi
atomic_positions=$(grep -e 'ATOMIC_POSITIONS (' $1 | tail -n 1)
cell_parameters=$(grep -e 'CELL_PARAMETERS (' $1 | tail -n 1)

#echo "Number of Atom"  $natom
#echo $na   $nc

if [ "$na" -eq "$nc" ]
then
#   echo 1 , vc-relax
    lines=$(grep -e 'ATOMIC_POSITIONS (' -A $natom  $1 | tail -n $natom)
    lattice=$(grep -e 'CELL_PARAMETERS (' -A 3  $1 | tail -n 3)
    sed -i "/ATOMIC_POSITIONS/,+${natom}d" $2
    sed -i "/CELL_PARAMETERS/,+3d" $2
    echo $cell_parameters >> $2
    echo "$lattice" >> $2
    echo $atomic_positions >> $2
    echo "$lines" >> $2
else
#   echo 2 , relax
    lines=$(grep -e 'ATOMIC_POSITIONS (' -A $natom  $1 | tail -n $natom)
    sed -i "/ATOMIC_POSITIONS/,+${natom}d" $2
    echo $atomic_positions >> $2
    echo "$lines" >> $2
fi
