use bindings_generat::enrichment::HeaderCommentParser;

fn main() {
    let header_content = r#"
/** Initialize a previously created tensor transform descriptor. */
CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                  const uint32_t nbDims,
                                  const cudnnTensorFormat_t destFormat,
                                  const int32_t padBeforeA[],
                                  const int32_t padAfterA[],
                                  const uint32_t foldA[],
                                  const cudnnFoldingDirection_t direction);
    "#;

    let parser = HeaderCommentParser::new().unwrap();
    let comments = parser.parse_header_content(header_content).unwrap();
    
    println!("Found {} comments", comments.len());
    for comment in comments {
        println!("Function: {}", comment.function_name);
        println!("Brief: {:?}", comment.brief);
        println!("Params: {} param docs", comment.param_docs.len());
    }
}
