FROM rocker/r-ver:4.3.1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff5-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libglpk-dev \
    make \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('tidyverse','caret','glmnet','randomForest','e1071','rpart','dbscan','isotree','factoextra','nnet','pROC','reshape2','kernlab'), repos='https://cloud.r-project.org/')"

COPY . .

RUN mkdir -p outputs models visuals

CMD ["Rscript", "app.R"]
