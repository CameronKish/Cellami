
import React, { useEffect, useRef } from 'react';
import Markdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';

const MarkdownViewer = ({ isOpen, onClose, content, highlightText, filename, variant = 'modal' }) => {
    const contentRef = useRef(null);

    // Helper to strip markdown syntax from text to match rendered HTML content
    const cleanMarkdown = (text) => {
        if (!text) return '';

        // Basic entity decoding
        const decodeEntities = (str) => {
            const textarea = document.createElement('textarea');
            textarea.innerHTML = str;
            return textarea.value;
        };

        let cleaned = text
            .replace(/^#+\s+/gm, '') // Remove headers
            .replace(/(\*\*|__)(.*?)\1/g, '$2') // Remove bold
            .replace(/(\*|_)(.*?)\1/g, '$2') // Remove italic
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links, keep text
            .replace(/!\[([^\]]*)\]\([^)]+\)/g, '') // Remove images
            .replace(/`([^`]+)`/g, '$1') // Remove inline code
            .replace(/^\s*[-*+]\s+/gm, '') // Remove list bullets
            .replace(/^\s*\d+\.\s+/gm, '') // Remove list numbers
            .replace(/<[^>]*>/g, '') // Remove HTML tags
            .replace(/^\s*\|.*\|\s*$/gm, (match) => { // Handle tables
                if (match.match(/^\s*\|[- :|]+\|\s*$/)) return ''; // Remove separator rows
                // Preserve escaped pipes
                let row = match.replace(/\\\|/g, '___PIPE___');
                row = row.replace(/\|/g, ' '); // Replace structure pipes with spaces
                return row.replace(/___PIPE___/g, '|'); // Restore content pipes
            })
            .replace(/\n/g, ' ') // Replace newlines with spaces
            .replace(/\s+/g, ' ') // Collapse whitespace
            .trim();

        return decodeEntities(cleaned);
    };

    useEffect(() => {
        if ((isOpen || variant === 'embedded') && highlightText && contentRef.current) {
            // Wait for render
            setTimeout(() => {
                const container = contentRef.current;
                const cleanedText = cleanMarkdown(highlightText);

                if (!cleanedText) return;

                // STRATEGY: Whitespace-agnostic matching
                // 1. Strip ALL whitespace from the search string.
                // 2. Build a "DOM String" by iterating text nodes and stripping their whitespace too,
                //    while keeping a map of (DOM String Index) -> (DOM Node, Original Offset).
                // 3. Match the stripped strings.
                // 4. Use the map to look up the start/end DOM positions.

                const searchString = cleanedText.replace(/\s/g, '');
                if (!searchString) return;

                const walker = document.createTreeWalker(
                    container,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );

                let currentNode = walker.nextNode();
                let domString = "";
                const nodeMap = []; // Maps index in domString to {node, indexInNode}

                while (currentNode) {
                    const text = currentNode.nodeValue;
                    for (let i = 0; i < text.length; i++) {
                        const char = text[i];
                        if (!/\s/.test(char)) { // If not whitespace
                            domString += char;
                            nodeMap.push({ node: currentNode, index: i });
                        }
                    }
                    currentNode = walker.nextNode();
                }

                console.log("Searching for (stripped):", searchString.substring(0, 50) + "...");
                // console.log("DOM String (stripped):", domString.substring(0, 200) + "...");

                const matchIndex = domString.indexOf(searchString);

                if (matchIndex !== -1) {
                    console.log("Match found at index:", matchIndex);

                    const startIndex = matchIndex;
                    const endIndex = matchIndex + searchString.length - 1; // Inclusive

                    if (startIndex < nodeMap.length && endIndex < nodeMap.length) {
                        const startInfo = nodeMap[startIndex];
                        const endInfo = nodeMap[endIndex];

                        const range = document.createRange();
                        range.setStart(startInfo.node, startInfo.index);
                        range.setEnd(endInfo.node, endInfo.index + 1); // +1 because setEnd is exclusive

                        const selection = window.getSelection();
                        selection.removeAllRanges();
                        selection.addRange(range);

                        // Scroll into view
                        startInfo.node.parentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                } else {
                    console.log("Could not find exact match.");
                    // Fallback: Try fuzzy match of first 20 chars
                    const fuzzyStart = searchString.substring(0, 20);
                    const fuzzyIndex = domString.indexOf(fuzzyStart);

                    if (fuzzyIndex !== -1) {
                        console.log("Found fuzzy start match");
                        const startInfo = nodeMap[fuzzyIndex];
                        startInfo.node.parentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });

                        // Select the fuzzy part
                        const endInfo = nodeMap[fuzzyIndex + fuzzyStart.length - 1];
                        const range = document.createRange();
                        range.setStart(startInfo.node, startInfo.index);
                        range.setEnd(endInfo.node, endInfo.index + 1);
                        const selection = window.getSelection();
                        selection.removeAllRanges();
                        selection.addRange(range);
                    }
                }

            }, 300); // Delay to ensure Markdown is rendered
        }
    }, [isOpen, highlightText, content, variant]);

    if (!isOpen && variant === 'modal') return null;

    const contentClass = "prose prose-slate max-w-none overflow-x-auto break-all whitespace-normal prose-pre:whitespace-pre-wrap prose-pre:break-all prose-code:break-all prose-headings:text-slate-800 prose-p:text-slate-600 prose-a:text-indigo-600 prose-strong:text-slate-800 prose-table:border-collapse prose-table:border prose-table:border-slate-300 prose-table:shadow-sm prose-th:border prose-th:border-slate-300 prose-th:bg-slate-100 prose-th:p-3 prose-th:text-slate-700 prose-td:border prose-td:border-slate-300 prose-td:p-3 prose-tr:even:bg-slate-50";

    if (variant === 'embedded') {
        return (
            <div className="h-full flex flex-col bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="flex-1 overflow-y-auto p-6 bg-white">
                    <div className={`${contentClass} w-full`} ref={contentRef}>
                        <Markdown
                            rehypePlugins={[rehypeRaw]}
                            remarkPlugins={[remarkGfm]}
                        >
                            {content}
                        </Markdown>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/75 backdrop-blur-sm animate-in fade-in duration-200">
            <div
                className="bg-slate-50 rounded-xl shadow-2xl w-[98%] h-[95vh] flex flex-col overflow-hidden animate-in zoom-in-95 duration-200 border border-slate-200 ring-1 ring-black/10"
                style={{ backgroundColor: '#f8fafc' }}
            >
                {/* Header */}
                <div className="flex items-center justify-between px-3 py-3 border-b border-slate-200 bg-white sticky top-0 z-10 shadow-sm shrink-0 min-w-0">
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                        <div className="p-2 bg-indigo-600 text-white rounded-lg shadow-md shrink-0">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                        </div>
                        <div className="min-w-0 flex-1">
                            <h3 className="font-bold text-slate-800 text-sm leading-tight truncate">{filename}</h3>
                            <p className="text-[10px] text-slate-500 font-medium uppercase tracking-wider hidden sm:block">Source Viewer</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="flex items-center gap-2 px-2 py-1.5 bg-white border border-slate-300 rounded-md shadow-sm hover:bg-slate-50 hover:border-slate-400 transition-all shrink-0 ml-2 mr-20"
                        style={{ marginRight: '30px' }}
                        title="Close Viewer"
                    >
                        <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">Close</span>
                        <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-4 md:p-8 bg-white min-w-0">
                    <div className={`${contentClass} w-full`} ref={contentRef}>
                        <Markdown
                            rehypePlugins={[rehypeRaw]}
                            remarkPlugins={[remarkGfm]}
                            components={{
                                pre: ({ node, ...props }) => <pre {...props} className="whitespace-pre-wrap break-all overflow-x-hidden" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }} />,
                                code: ({ node, ...props }) => <code {...props} className="whitespace-pre-wrap break-all" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }} />
                            }}
                        >
                            {content}
                        </Markdown>
                    </div>
                </div>

                {/* Footer */}
                <div className="px-4 py-3 bg-slate-50 border-t border-slate-100 text-xs text-slate-400 flex justify-between items-center shrink-0 min-w-0">
                    <span className="truncate mr-2">Markdown View</span>
                    <span className="whitespace-nowrap">{content ? content.length.toLocaleString() : 0} chars</span>
                </div>
            </div>
        </div>
    );
};

export default MarkdownViewer;
