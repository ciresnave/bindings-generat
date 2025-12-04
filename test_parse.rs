use bindings_generat::enrichment::header_parser::HeaderCommentParser;
fn main() {
    let content = r#"
/**
 * @brief Creates a handle
 * 
 * @details This function initializes a new handle that must be
 * used for all subsequent operations. The handle maintains internal state.
 * 
 * @param[out] handle Pointer to handle
 * @return SUCCESS or ERROR
 */
void create(void** handle);
"#;
    let parser = HeaderCommentParser::new().unwrap();
    let comments = parser.parse_header_content(content).unwrap();
    println!("Found {} comments", comments.len());
    for comment in comments {
        println!("Function: {}", comment.function_name);
        println!("Brief: {:?}", comment.brief);
        println!("Detailed: {:?}", comment.detailed);
    }
}
