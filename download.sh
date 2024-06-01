counter=2000
while [ $counter -le 2384 ]
    do
    wget -q https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Female-Images/70mm/4K_Tiff-Images/${counter}.tif
    mv ${counter}.tif data/abdomen/
    echo $counter
    sleep 3
    ((counter++))
done
