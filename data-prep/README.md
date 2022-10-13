# Data acquisition and preparation

1) Download and uncompress the `stackoverflow.com-Posts.7z` dataset from the [Internet Archive](https://archive.org/download/stackexchange). The uncompressed data is around 100GB. 
2) Move the dataset, `Posts.xml`, to the `data` folder.
3) Create a small sample of the dataset called `1000Posts.xml` using the following command: `head -n 1002 Posts.xml | sed 's/$/\n<\/posts>/' > 1000Posts.xml`
4) Execute [process-raw-xml.py](process-raw-xml.py) using the [`environment-data-prep`](../environment-data-prep.xml) conda environment.

You'll find the output files in the data-prep folder. There are three folders you care about (the rest are temporary folders and safe to delete):

 * `data/qna.parquet`: Every python-related question and answer.
 * `data/questions-python-only.parquet`: Python-related questions along with their accepted answer or, in case of a missing accepted answer, their highest-scored answer.
 * `data/questions-python-only-multipart.parquet`: Same as `data/questions-python-only.parquet`, but the data is split over multiple files.



