
# The script assumes each line of invalid_xml.tsv starts with the PMCID.
# It does not rely on the 2nd column path, so minor format differences won’t break it.
# Always keep a backup or use the move-to-backup version first if you’re uncertain.

# This only prints the rm commands — check the list carefully before running.
cd 3.xml/
# awk '{print "rm -v " $1 ".xml"}' invalid_xml.tsv

# Actual deletion version
# Once verified the output, run:
# awk '{system("rm -v " $1 ".xml")}' invalid_xml.tsv

# move to a quarantine folder instead of deleting
mkdir -p invalid_backup
# awk '{system("mv -v " $1 ".xml invalid_backup/")}' invalid_xml.tsv