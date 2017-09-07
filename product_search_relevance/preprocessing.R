library(data.table)
library(stringdist)
library(ngram)
library(e1071)


main <- function()
{
  
  #read product descriptions 
  print("reading product descriptions");
  product_descriptions <- data.table( read.csv("product_descriptions.csv",stringsAsFactors = FALSE,fileEncoding = "latin1"));
  #read product attributes
  print("reading product attributes");
  productAttributes <- data.table(read.csv("attributes.csv",stringsAsFactors = FALSE,fileEncoding = "latin1"));
  #read the train set
  print("reading the train set")
  trainSet <- data.table( read.csv("train.csv",stringsAsFactors = FALSE,fileEncoding = "latin1") );
  
  #subset trainset just for testing
  trainSet <- trainSet[1:2000,];
  
  ######do cleaning and standarization#####
  #add a rownumber to the datatase
  trainSet[, rownum:=.I];
  #convert all strings to lower case
  print("Converting all strings to lower case");
  trainSet$product_title = tolower(trainSet$product_title);
  trainSet$search_term = tolower(trainSet$search_term);
  
  product_descriptions$product_description = tolower(product_descriptions$product_description);
  
  productAttributes$name = tolower(productAttributes$name);
  productAttributes$value = tolower(productAttributes$value);
  
  #Remove tildes
  print("Removing tildes");
  trainSet$product_title = iconv(trainSet$product_title,to="ASCII//TRANSLIT");
  trainSet$search_term = iconv(trainSet$search_term,to="ASCII//TRANSLIT");
  
  product_descriptions$product_description = iconv(product_descriptions$product_description,to="ASCII//TRANSLIT");
  
  productAttributes$name = iconv(productAttributes$name,to="ASCII//TRANSLIT");
  productAttributes$value = iconv(productAttributes$value,to="ASCII//TRANSLIT");
  
  ##join train set with descriptions
  setkey(product_descriptions,product_uid);
  setkey(trainSet,product_uid);
  print("merging trainset with descriptions");
  trainSet <- merge(trainSet,product_descriptions);
  
  #Vectorize functions to apply to data table
  vectorized_get_ngrams <- Vectorize(get_ngrams);
  vectorized_get_ngrams_from_vector <- Vectorize(get_ngrams_from_vector)
  vectorized_is_digit <- Vectorize(is_digit);
  
  ######prepare the dataset for machine learning(feature engineering)#######
  #Get query, title and description unigrams
  print("Getting query,title and description unigrams");
  trainSet[,query_unigrams:=vectorized_get_ngrams(search_term,1)];
  trainSet[,title_unigrams:= vectorized_get_ngrams(product_title,1)];
  trainSet[,description_unigrams:=vectorized_get_ngrams(product_description,1)];
  print("Getting query,title and description bigrams");
  trainSet[,query_bigrams := vectorized_get_ngrams(search_term,2)];
  trainSet[,title_bigrams := vectorized_get_ngrams(product_title,2)];
  trainSet[,description_bigrams := vectorized_get_ngrams(product_description,2)];
  print("Getting query ,title and description trigrams")
  trainSet[,query_trigrams := vectorized_get_ngrams(search_term,3)];
  trainSet[,title_trigrams := vectorized_get_ngrams(product_title,3)];
  trainSet[,description_trigrams := vectorized_get_ngrams(product_description,3)];
  
  #simple counting  features(count number of ngrams)
  print("Generating simple ngram counts");
  trainSet[,query_unigrams_count := sapply(query_unigrams,length)];
  trainSet[,title_unigrams_count := sapply(title_unigrams,length)];
  trainSet[,description_unigrams_count := sapply(description_unigrams,length)];
  
  trainSet[,query_bigrams_count := sapply(query_bigrams,length)];
  trainSet[,title_bigrams_count := sapply(title_bigrams,length)];
  trainSet[,description_bigrams_count := sapply(description_bigrams,length)];
  
  trainSet[,query_trigrams_count := sapply(query_trigrams,length)];
  trainSet[,title_trigrams_count := sapply(title_trigrams,length)];
  trainSet[,description_trigrams_count := sapply(description_trigrams,length)];
  
  print("Generating digits counts and ratios");
  trainSet[,query_digit_count := sapply(query_unigrams,function(unigrams_list){sum(vectorized_is_digit(unigrams_list))})];
  trainSet[,title_digit_count := sapply(title_unigrams,function(unigrams_list){sum(vectorized_is_digit(unigrams_list))})];
  trainSet[,description_digit_count := sapply(description_unigrams,function(unigrams_list){sum(vectorized_is_digit(unigrams_list))})];
  
  trainSet[,query_digit_ratios := query_digit_count/query_unigrams_count];
  trainSet[,title_digit_ratios := title_digit_count/title_unigrams_count];
  trainSet[,description_digit_ratios := description_digit_count/description_unigrams_count];
  
  print("Generating intersection word counts")
  #Unigrams
  print("##Generating unigram instersection counts")
  trainSet$query_in_title_unigram_count<- apply(trainSet,1,function(row){ngram_insersection_count(row[["query_unigrams"]],row[["title_unigrams"]])});
  trainSet$query_in_description_unigram_count<- apply(trainSet,1,function(row){ngram_insersection_count(row[["query_unigrams"]],row[["description_unigrams"]])});
  #Bigrams
  print("##Generating bigram intersection counts")
  trainSet$query_in_title_bigram_count<- apply(trainSet,1,function(row){ngram_insersection_count(row[["query_bigrams"]],row[["title_bigrams"]])});
  trainSet$query_in_description_bigram_count<- apply(trainSet,1,function(row){ngram_insersection_count(row[["query_bigrams"]],row[["description_bigrams"]])});
  #Trigrams
  print("##Generating trigram intersection counts")
  trainSet$query_in_title_trigram_count<- apply(trainSet,1,function(row){ngram_insersection_count(row[["query_trigrams"]],row[["title_trigrams"]])});
  trainSet$query_in_description_trigram_count<- apply(trainSet,1,function(row){ngram_insersection_count(row[["query_trigrams"]],row[["description_trigrams"]])});
  
  #Distance features
  ##JaccardCoef(size of intersection / size of union)
  print("Generating jaccard coefficient betweeen query and title")
  trainSet$query_title_intersection_length<- apply(trainSet,1,function(row){length(intersect(row["query_unigrams"][[1]],row["title_unigrams"][[1]]))})
  trainSet$query_title_union_length<- apply(trainSet,1,function(row){length(union(row["query_unigrams"][[1]],row["title_unigrams"][[1]]))})
  trainSet[,query_and_title_jaccard_dist := query_title_intersection_length/query_title_union_length]
  
  print("Generaton jaccard coefficiente  between query and description")
  trainSet$query_description_intersection_length<- apply(trainSet,1,function(row){length(intersect(row["query_unigrams"][[1]],row["description_unigrams"][[1]]))})
  trainSet$query_description_union_length<- apply(trainSet,1,function(row){length(union(row["query_unigrams"][[1]],row["description_unigrams"][[1]]))})
  trainSet[,query_and_description_jaccard_dist := query_description_intersection_length/query_description_union_length]
  
  print("Generating dice distance");
  trainSet[,query_and_title_dice_dist := (2*query_title_intersection_length)/(query_unigrams_count + title_unigrams_count)];
  trainSet[,query_and_description_dice_dist := (2*query_description_intersection_length)/(query_unigrams_count + description_unigrams_count)];
  #feature 3: number of words in search query that exists in product attributes
  #print("calculating the number of words in query that exists in product ttributes")
  #trainSet <- trainSet[,query_words_in_product_attributes:=query_words_in_product_attributes_count(product_uid,search_term,productAttributes),by=rownum];
  #feature 4 and 5: "string distance" between search term and product title,product description
  trainSet <- trainSet[, string_distance_search_and_product_title :=stringdist(search_term,product_title)];
  trainSet[,string_distance_search_and_product_description := stringdist(search_term,product_description)];
  
  
  ####Read the testset
  #testSet <- data.table(read.csv("test.csv",stringsAsFactors = FALSE));
  
  
  return (trainSet);
}

#function that counts how many words in the search query exists in the product title
#the train set is pased as parameter
query_words_in_product_title_and_description_count <- function(trainSet)
{
  
  print("Calculating the number of words of query that exists in product title and description")
  #trainSet[,query_words_in_product_title_count:={sum(sapply(strsplit(search_term," ")[[1]],string_contains,containing_string=product_title))}];
  trainSet[,query_words_in_product_title_count:={search_words<-get_unigrams(search_term);containing_string<-product_title;sum(sapply(search_words,string_contains,containing_string))},by=rownum];
  trainSet[,query_words_in_product_description_count:={search_words<-get_unigrams(search_term);containing_string<-product_description;sum(sapply(search_words,string_contains,containing_string))},by=rownum]
  
  return (trainSet);
}

##function that counts how many words in the search query exists in product attributes
query_words_in_product_attributes_count <- function(productId,search_term,productAttributes)
{

   #filter only attributes for the given product id
  filteredProductAttributes<- productAttributes[product_uid==productId];
  #add rownumber to the attributes
  filteredProductAttributes[,rownum:=.I];
  
  filteredProductAttributes[,word_count := sum(sapply(get_unigrams(search_term),string_contains,containing_string=value)),by=rownum];
  
  return(filteredProductAttributes[,sum(word_count)]);
}

#auxiliar function that returns 1 if a string is contained inside other string(passed as parameters)
#returns 0 otherwise
string_contains <- function(contained_string, containing_string)
{

  if(length(grep(contained_string, containing_string))>0){ 
    return (1);
  }
  else {
    return (0);
  }
}

#auxiliar function that gets a text and separates them in single words(unigrams)
get_unigrams <- function(text)
{
  words <- ngram(text,n=1);
  words <- get.ngrams(words);
  
  return (words);
}

bigram_count <- function(query,title)
{
  bigram <- ngram(query,n=2);
  result <- 0;
  
  for(index in 1:length(get.ngrams(bigram)))
  {
    result <- result + string_contains(get.ngrams(bigram)[index],title);
  }
  
  return (result);
}


get_ngrams<-function(text,n)
{
  result <- tryCatch(
    {
      words <- ngram(text,n=n);
      words <- get.ngrams(words);
      
      return(words);
    },
    error=function(cond)
    {
      return (NULL);
    },
    warning=function(cond){
      return (NULL);
    }
  )
  

  return (result);
}

get_ngrams_from_vector <- function(text_vector,n)
{
  if(length(text_vector)>= n){
    return (get_ngrams(concat(text_vector,collapse = " "),n));
  }
  else
  {
    return (NA);
  }

}
get_ngram_data_table<-function(text,n)
{
  ngram_name <- paste0(n,"gram");
  new_table <- data.frame(ngram_name=get_ngrams(text,n));
  colnames(new_table)<-ngram_name;
  #new_table[,ngram_name] <- get_ngrams(text,n);
  return (data.table(new_table));
}


is_digit<- function(N){
  !grepl("[^[:digit:]]", format(N,  digits = 20, scientific = FALSE))
}

#function that counts how many ngrams in a apperas in the ngrams in b
#for example "hola mundo cruel" and "hola mundo bonito" return 2
 ngram_insersection_count <- function(ngram_a,ngram_b)
{
   result <- 0;
  if(!is.null( ngram_a) & ! is.null( ngram_b ))
    result <- sum(sapply(ngram_a, function(ngram){ngram==ngram_b}));

  return (result);
 }

