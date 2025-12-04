use regex::Regex;
use std::collections::HashMap;

fn main() {
    let inline_comment_re = Regex::new(r"/\*\s*(.+?)\s*\*/").unwrap();

    let test_line = "    int c);                   /* number of channels */";

    println!("Testing line: {:?}", test_line);

    if let Some(comment_match) = inline_comment_re.captures(test_line) {
        let comment_text = comment_match
            .get(1)
            .map(|m| m.as_str())
            .unwrap_or("")
            .trim();
        let before_comment = &test_line[..test_line.find("/*").unwrap_or(test_line.len())];

        println!("Found comment: {:?}", comment_text);
        println!("Before comment: {:?}", before_comment);

        // Now extract param name
        let cleaned = before_comment
            .trim()
            .trim_end_matches(',')
            .trim_end_matches(')')
            .trim_end_matches(';')
            .trim();

        println!("After cleaning: {:?}", cleaned);

        let parts: Vec<&str> = cleaned.split_whitespace().collect();
        println!("Parts: {:?}", parts);

        if let Some(last) = parts.last() {
            let param = last.trim_start_matches('*').trim_start_matches('&').trim();
            println!("Param name: {:?}", param);
        }
    } else {
        println!("NO MATCH!");
    }
}
