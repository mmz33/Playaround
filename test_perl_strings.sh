#!/bin/bash

str1="hello #i" # normal string
str2="hello #ă" # some random unicode character

# want to convert #.. to (#..)

echo $str1 | perl -pe 's/(#[\w]*)+/\(\1\)/g'
echo $str2 | perl -pe 's/(#[\w]*[^\x00-\x7F]*)+/\(\1\)/g'
