prompt = f"""Task: Generate OpenSearch Query

                        [Input]
                        - OpenSearch Mapping: {mapping}
                        - User Query: {query} 
                        - Main Category: {cate}
                        - Sub Categories: {sub_categories}

                        [Note]
                        - Ensure selecting best sub category exclusively from provided lists. 
                        - Only print the Query nothing else no explanation and opening text etc. 
                        - Please refrain from employing nested queries.
                        - Kindly ensure that your query should align with the Main Categories and Sub Categories listed above. Also don't use more than 2 categories in the generated query.
                        - Kindly abstain from modifying types categorized as 'keyword' in the respective type, maintain the original format of entries. This guideline extends to all type classified as 'keyword' in mapping.
                        - Also ensure to strictly follow the OpenSearch Mapping.
                        - Incorporate keywords like "most popular," "famous," or "top" into the search query to prioritize highly-rated items.
                        [Schema]
                        - Categories should be filtered inside (filter clause)
                        - for any Range or sorting, number of rating, actual price, discounted price, or any other integer fields should be inside (Should clause)
                        - Name should be inside must/match clause.
                        - Use term query not terms query.
                        """