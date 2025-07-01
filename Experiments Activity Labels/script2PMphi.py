#Imports
import networkx as nx
import csv
import random
import matplotlib.pyplot as plt
import re
import os
from huggingface_hub import login
from datetime import datetime
import pm4py
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization


#Huggingface Login
login(token="HERE_YOUR_TOKEN")
print("Login successfull")

processGraph = nx.DiGraph()

log = pm4py.read_xes("top_10_variants_BPI2017.xes")

dfg_perf = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)

# Create DFG
for (source, target), duration in dfg_perf.items():
    processGraph.add_edge(source, target, weight=round(duration, 3))



def describe_graph(graph):
    nodes = graph.nodes()
    description = f"The directed process graph contains the nodes: {', '.join([f'{n}' for n in nodes])}. "
    for u, v, data in graph.edges(data=True):
        description += f"There is an edge from {u} to {v} with duration {data['weight']}. "
    return description


word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10
        }


def normalize_answer(answer, question,graph,questiondID):
    answer = str(answer).lower()
    graph_nodes_lower = {str(n).lower() for n in graph.nodes}

    if "Y-N" in questiondID:
        has_yes = re.search(r'\b(yes|true)\b', answer.lower())
        has_no = re.search(r'\b(no|false)\b', answer.lower())
        if has_yes and has_no:
            return "Unclear"
        elif has_yes:
            return "Yes"
        elif has_no:
            return "No"
        
    elif "length of the shortest simple path" in question:
        match = re.search(
            r"(?i).*?(?:length of the shortest (?:simple )?path is|"
            r"length of this path is|"
            r"the answer is|"
            r"length of the path is|"
            r"length is|"
            r"shortest (?:simple )?path from [^\d]+ to [^\d]+ is)\s+(-?\d+)",
            answer
        )
        if match:
            return int(match.group(1))
        else:
            fallback_match = re.search(r"(-?\d+)", answer)
            if fallback_match:
                return int(fallback_match.group(0))
            
    elif "length of the longest simple path" in question:
        match = re.search(
            r"(?i).*?(?:length of the longest (?:simple )?path is|"
            r"length of this path is|"
            r"the answer is|"
            r"length of the path is|"
            r"length is|"
            r"longest (?:simple )?path from [^\d]+ to [^\d]+ is)\s+(-?\d+)",
            answer
        )
        if match:
            return int(match.group(1))
        else:
            fallback_match = re.search(r"(-?\d+)", answer)
            if fallback_match:
                return int(fallback_match.group(0))
            
    elif "how many edges" in question.lower():
        match = re.search(
            r'(?i)\b(-?\d+)\s+edges\b(?:\s|[.,])?',
            answer
        )
        if not match:
            match = re.search(
                r'(?i)\b(-?\d+)\s+edge\b(?:\s|[.,])?',
                answer
            )

        if match:
            return int(match.group(1))
        else:
            fallback_match = re.search(r"(-?\d+)", answer)
            if fallback_match:
                return int(fallback_match.group(0))
            
    elif "how many nodes" in question.lower():
        match = re.search(
            r'(?i)\b(-?\d+)\s+(?:nodes|vertices)\b(?:\s|[.,])?',
            answer
        )
        if not match:
            match = re.search(
                r'(?i)\b(-?\d+)\s+(?:node|vertice)\b(?:\s|[.,])?',
                answer
            )

        if match:
            return int(match.group(1))
        else:
            fallback_match = re.search(r"(-?\d+)", answer)
            if fallback_match:
                return int(fallback_match.group(0))
            
    elif "how many simple cycles contains" in question.lower():
        match = re.search(
            r"(?i).*?(?:answer|number of cycles|number of\s+\w+)\s+is\s+(-?\d+)(?![-,])",
            answer
        )

        if not match:
            match = re.search(
                r'(?i)\b(?:has|contains)\s+(-?\d+)\s+cycles\b[.,]?',
                answer
            )

        if match:
            return int(match.group(1))  
        
        elif any(phrase in answer.lower() for phrase in [
        "the graph is acyclic",
        "the graph has no simple cycles",
        "the graph has no cycles"
        ]):
            return 0
        else:
            match_word = re.search(
            r"(?i)\b(" + "|".join(word_to_num.keys()) + r")\b\s+(?:simple\s+)?cycle[s]?\b[.,]?",
            answer
        )

        if match_word:
            return word_to_num[match_word.group(1).lower()]
        else:
            fallback_match = re.search(r"(-?\d+)", answer)
            if fallback_match:
                return int(fallback_match.group(0))
            
    elif "how many" in question.lower() or "What is the highest degree of all nodes in the graph" in question:
        match = re.search(
            r"(?i).*?(?:highest degree(?: of all nodes in the graph)?|answer|number of cycles|number of\s+\w+)\s+is\s+(-?\d+)(?![-,])",
            answer
        )
        if match:
            return int(match.group(1))
        else:
            fallback_match = re.search(r"(-?\d+)", answer)
            if fallback_match:
                return int(fallback_match.group(0))

    elif "What are the names of the nodes with the highest degree" in question or "Which nodes have no" in question:
        if "none" in answer.lower():
            return "None"
        
        if answer== "[]":
            return "None"
        
        text = re.sub(r'\band\b', ',', answer, flags=re.IGNORECASE)

        #remove brackets
        text = re.sub(r'\(\d+\)', '', text)

        #seperated by commas
        comma_matches = re.findall(r'((?:\w+\s*,\s*)+\w+)', text)
        if comma_matches:
            best = max(comma_matches, key=lambda m: len(re.split(r'\s*,\s*', m)))
            parts = re.split(r'\s*,\s*', best.strip())
            parts = [p for p in parts if p.lower() in graph_nodes_lower]
            return ','.join(parts)

        #sperated by empty spaces
        word_matches = re.findall(r'((?:\w+\s+){1,}\w+)', text)
        if word_matches:
            best = max(word_matches, key=lambda m: len(m.strip().split()))
            parts = best.strip().split()
            parts = [p for p in parts if p.lower() in graph_nodes_lower]
            return ','.join(parts)

        #Single value found?
        fallback = re.findall(r'\b\w+\b', text)
        if fallback:
            return ','.join(fallback)
        
    elif "What is the longest simple path" in question or "What is the shortest simple path" in question:
 
        if "no path" in answer.lower():
            return "No path exists"
        
        # Search Paths (A->B->C)
        arrow_matches = re.findall(r'((?:\w+\s*->\s*)+\w+)', answer)
        if arrow_matches:
            best = max(arrow_matches, key=lambda m: len(re.split(r'\s*->\s*', m)))
            parts = re.split(r'\s*->\s*', best.strip())
            parts = [p for p in parts if p.lower() in graph_nodes_lower]
            return ','.join(parts)


        # Search Paths (A,B,C)
        comma_matches = re.findall(r'((?:\w+\s*,\s*)+\w+)', answer)
        if comma_matches:
            best = max(comma_matches, key=lambda m: len(re.split(r'\s*,\s*', m)))
            parts = re.split(r'\s*,\s*', best.strip())
            parts = [p for p in parts if p.lower() in graph_nodes_lower]
            return ','.join(parts)

        # Search Paths (A-B-C)
        dash_matches = re.findall(r'((?:\w+\s*-\s*)+\w+)', answer)
        if dash_matches:
            best = max(dash_matches, key=lambda m: len(re.split(r'\s*-\s*', m)))
            parts = re.split(r'\s*-\s*', best.strip())
            parts = [p for p in parts if p.lower() in graph_nodes_lower]
            return ','.join(parts)

        # Search Paths (A B C)
        space_matches = re.findall(r'((?:\w+\s+)+\w+)', answer)
        if space_matches:
            best = max(space_matches, key=lambda m: len(m.strip().split()))
            parts = best.strip().split()
            parts = [p for p in parts if p.lower() in graph_nodes_lower]
            return ','.join(parts)
      
    return "Normalizing not possible"


def normalize_set_string(s):
    return sorted([item.strip().lower() for item in s.split(',') if item.strip()])


def evaluate_responses(graph, question, answer,ground_truth,questionID):
    evaluation_results = {}
    ground_truth_to_compare = ground_truth
    normalized = normalize_answer(str(answer), question,graph,questiondID)

    if "Shortest path - Path" in questionID:
        if normalized.strip().lower() and normalized.strip().lower() != "no path exists":
            try:
                normalized_list = [x.strip().lower() for x in normalized.split(',')]
                if isinstance(ground_truth, list) and all(isinstance(p, list) for p in ground_truth):
                    ground_truth_lowered = [[str(x).strip().lower() for x in path] for path in ground_truth]
                    if normalized_list in ground_truth_lowered:
                        ground_truth_to_compare = normalized
                else:
                    ground_truth_single = [str(x).strip().lower() for x in ground_truth]
                    if normalized_list == ground_truth_single:
                        ground_truth_to_compare = normalized
            except:
                print("Error in Evaluate_response")

    if "Longest path - Path" in questionID:
        if normalized.strip().lower() and normalized.strip().lower() != "no path exists":
            try:
                normalized_list = [x.strip().lower() for x in normalized.split(',')]
                ground_truth_single = [str(x).strip().lower() for x in ground_truth]
                if normalized_list == ground_truth_single:
                    ground_truth_to_compare = normalized
            except:
                print("Error in Evaluate_response")

    if "Highest degree node" in questionID or "No successors" in questionID or "No predecessors" in questionID:
        norm_set = normalize_set_string(normalized)
        truth_set = normalize_set_string(ground_truth)
        if  norm_set == truth_set:
                ground_truth_to_compare=normalized

    evaluation_results["x"] = {
        "Antwort": answer,
        "Normalisiert": normalized,
        "Korrekt": str(normalized) == str(ground_truth_to_compare) 
    }

    return evaluation_results

def computeGroundTruth(question, graph, selected_nodes, selected_edges):
    
    if "How many nodes contains the graph?" in question:
        ground_truth = len(graph.nodes)
    elif "How many edges contains the graph?" in question:
        ground_truth = len(graph.edges)
    elif "Does the graph contain any cycles?" in question:
        ground_truth = "Yes" if nx.is_directed_acyclic_graph(graph) == False else "No"
    elif "Exists a node called" in question:
        ground_truth = "Yes" if selected_nodes[0] in graph.nodes else "No"
    elif "How many nodes can be reached directly from node" in question:
        ground_truth = len(list(graph.neighbors(selected_nodes[0])))
    elif "How many nodes are reachable from node" in question:
        if nx.is_directed(graph):
            ground_truth = len(nx.descendants(graph, selected_nodes[0]))
        else:
            ground_truth = len(nx.node_connected_component(graph, selected_nodes[0]))
    elif "Does node" in question and "directly follow node" in question:
        ground_truth = "Yes" if (selected_nodes[1], selected_nodes[0]) in graph.edges else "No"
    elif "Is node" in question and "a direct successor of node" in question:
        ground_truth = "Yes" if (selected_nodes[1], selected_nodes[0]) in graph.edges else "No"
    elif "What are the names of the nodes with the highest degree?" in question:
        max_degree = max(graph.degree, key=lambda x: x[1])[1]
        highest_degree_nodes = [node for node, degree in graph.degree if degree == max_degree]
        ground_truth = ",".join(map(str, highest_degree_nodes))
    elif "What is the highest degree of all nodes in the graph?" in question:
        return max(graph.degree, key=lambda x: x[1])[1]
    elif "How many simple cycles contains the graph?" in question:
        return len(list(nx.simple_cycles(graph))) if nx.is_directed(graph) else "Not implemented for undirected graphs"
    elif "Which simple cycles contains the graph" in question:
        return list(nx.simple_cycles(graph)) if nx.is_directed(graph) else "Not implemented for undirected graphs"
    elif "Are node" in question and "concurrent?" in question:
        return "Yes" if selected_nodes[0] not in nx.ancestors(graph, selected_nodes[1]) and selected_nodes[1] not in nx.ancestors(graph, selected_nodes[0]) else "No"
    elif "Is there a Hamilton path in the graph?" in question:
        return "Yes" if nx.is_semiconnected(graph) else "No"
    elif "How many different simple paths are in the graph from node" in question:
        return len(list(nx.all_simple_paths(graph, source=selected_nodes[0], target=selected_nodes[1])))
    elif "What are all possible simple paths from node" in question:
        return list(nx.all_simple_paths(graph, source=selected_nodes[0], target=selected_nodes[1]))
    elif "Which nodes have no successors?" in question:
        ground_truth = [node for node in graph.nodes if graph.out_degree(node) == 0]
        if not ground_truth:
            ground_truth = "None"
        else:
            ground_truth = ",".join(map(str, ground_truth))
    elif "Is node" in question and "a direct predecessor of node" in question:
        ground_truth = "Yes" if (selected_nodes[0], selected_nodes[1]) in graph.edges else "No"
    elif "From how many nodes is node" in question and "directly reachable?" in question:
        ground_truth = len([node for node in graph.predecessors(selected_nodes[0])])
    elif "From how many nodes is node" in question and "reachable?" in question:
        if nx.is_directed(graph):
            ground_truth = len(nx.ancestors(graph, selected_nodes[0]))
        else:
            ground_truth = len(nx.node_connected_component(graph, selected_nodes[0]))
    elif "Which nodes have no predecessor?" in question:
        ground_truth = [node for node in graph.nodes if graph.in_degree(node) == 0]
        if not ground_truth:
            ground_truth = "None"
        else:
            ground_truth = ",".join(map(str, ground_truth))
    elif "What is the length of the shortest simple path from node" in question:
        try:
            ground_truth = nx.shortest_path_length(graph, source=selected_nodes[0], target=selected_nodes[1], weight='weight')
        except nx.NetworkXNoPath:
            ground_truth = "No path exists"
    elif "What is the shortest simple path from node" in question:
        try:
            paths = list(nx.all_shortest_paths(graph, source=selected_nodes[0], target=selected_nodes[1], weight='weight'))
            ground_truth = paths   
        except nx.NetworkXNoPath:
            ground_truth = "No path exists"
    elif "What is the length of the longest simple path from node" in question:
        try:
            paths = list(nx.all_simple_paths(graph, source=selected_nodes[0], target=selected_nodes[1]))
            ground_truth = max(sum(graph[u][v]['weight'] for u, v in zip(path, path[1:])) for path in paths)  
        except ValueError:
            ground_truth = "No path exists"
    elif "What is the longest simple path from node" in question:
        try:
            paths = list(nx.all_simple_paths(graph, source=selected_nodes[0], target=selected_nodes[1]))
            ground_truth = max(paths, key=lambda path: sum(graph[u][v]['weight'] for u, v in zip(path, path[1:])))
        except ValueError:
            ground_truth = "No path exists"
    elif "Is there a path from node" in question:
        ground_truth = "Yes" if nx.has_path(graph, selected_nodes[0], selected_nodes[1]) else "No"
    elif "What is the path from node" in question:
        try:
            ground_truth = nx.shortest_path(graph, source=selected_nodes[0], target=selected_nodes[1])
            ground_truth = ",".join(map(str, ground_truth))
        except nx.NetworkXNoPath:
            ground_truth = "No path exists"
    else:
        ground_truth = "Not implemented for this question"
        
    return ground_truth




def getQuestionForIndividualGraph(question,questiondID,graph):
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    other_activities = ["A_Registered", "A_Reviewed", "A_Approved", "A_Rejected", "A_Updated", "A_Archived", "W_Assess application", "W_Contact customer", "W_Review documents", "W_Evaluate request", "W_Request missing info", "O_Send contract", "O_Receive documents", "O_Verify identity", "O_Close case", "O_Forward to department", "O_Update status", "O_Escalated", "A_Scheduled", "W_Final approval"]


    path_questions = [
        "Length of shortest path - Number",
        "Shortest path - Path",
        "Length of longest path - Number",
        "Longest path - Path",
        "Path variants count - Number"
    ]
    
    if "Exists a node called <node>?" in question:
        available_nodes = nodes + other_activities 
    else:
        available_nodes = nodes
    
    if "<node>" in question:
        num_nodes_needed = question.count("<node>")
        if questiondID in path_questions and num_nodes_needed == 2:
            valid_pairs = [
                (u, v) for u in nodes for v in nodes
                if u != v and nx.has_path(graph, u, v)
            ]
            if valid_pairs:
                selected_nodes = list(random.choice(valid_pairs))
            else:
                selected_nodes = random.sample(nodes, 2)
        else:
            selected_nodes = random.sample(available_nodes, min(num_nodes_needed, len(nodes)))
    else:
        selected_nodes = []
    
    if "<edge>" in question:
        num_edges_needed = question.count("<edge>")
        selected_edges = random.sample(edges, min(num_edges_needed, len(edges)))
    else:
        selected_edges = []

    for node in selected_nodes:
        question = question.replace("<node>", str(node), 1)
    for edge in selected_edges:
        edge_str = f"({edge[0]}, {edge[1]})" 
        question = question.replace("<edge>", edge_str, 1)
    
    ground_truth= computeGroundTruth(question, graph, selected_nodes, selected_edges)
    
    return question, ground_truth



#Create all lists
connectivityQuestions = [
    ("Directly follow - Y-N",                   "Does node <node> directly follow node <node>? "),
    ("Direct successor - Y-N",                  "Is node <node> a direct successor of node <node>? "),
    ("Directly reachable nodes - Number",       "How many nodes can be reached directly from node <node>? "),
    ("Reachable nodes - Number",                "How many nodes are reachable from node <node>? "),
    ("No successors",                           "Which nodes have no successors? "),
    ("Direct predecessor - Y-N",                "Is node <node> a direct predecessor of node <node>? "),
    ("Directly reachable from nodes - Number",  "From how many nodes is node <node> directly reachable? "),
    ("Reachable from nodes - Number",           "From how many nodes is node <node> reachable? "),
    ("No predecessors",                          "Which nodes have no predecessor? ")
] 
metricsQuestions = [
    ("Node existence - Y-N",           "Exists a node called <node>? "),  
    ("Number of nodes - Number",       "How many nodes contains the graph? "),
    ("Number of edges - Number",       "How many edges contains the graph? "),
    ("Highest degree node - Node",     "What are the names of the nodes with the highest degree? "),
    ("Highest degree - Number",        "What is the highest degree of all nodes in the graph? ")
]
topologyQuestions = [    
    ("Cycle existence - Y-N",           "Does the graph contain any cycles? "),
    ("Number of Cycles - Number",        "How many simple cycles contains the graph? "),
    ("Concurrency Check - Y-N",          "Are node <node> and <node> concurrent? "),
    ("Hamilton path existence - Y-N",    "Is there a Hamilton path in the graph? ")
]
pathandreachabilityQuestions = [
    ("Length of shortest path - Number",   "What is the length of the shortest simple path from node <node> to node <node>? "),
    ("Shortest path - Path",               "What is the shortest simple path from node <node> to node <node>? "),
    ("Length of longest path - Number",    "What is the length of the longest simple path from node <node> to node <node>? "),
    ("Longest path - Path",                "What is the longest simple path from node <node> to node <node>? "),
    ("Path existence - Y-N",               "Is there a path from node <node> to node <node>? "),
    ("Path variants count - Number",       "How many different simple paths are in the graph from node <node> to node <node>? ")
]
questions_map = {
    "Topology": topologyQuestions,
    "Connectivity": connectivityQuestions,
    "Path+Reachability": pathandreachabilityQuestions,
    "Metrics": metricsQuestions
}
all_questions_without_lists = topologyQuestions + connectivityQuestions + metricsQuestions + pathandreachabilityQuestions 

filteredForQuestionID = [tup[0] for tup in all_questions_without_lists]


promptings = ["zs", "fs","zcot", "bag", "algo", "sc", "role", 
              "fs+zcot", "bag+fs", "fs+role", "algo+fs", "bag+zcot", "role+zcot","algo+bag", "bag+role", "algo+role", "role+sc", "fs+sc","algo+sc","zcot+sc","bag+sc",
              "bag+fs+zcot", "algo+bag+fs", "fs+zcot+sc", "algo+fs+sc", "fs+role+zcot", "algo+fs+role","bag+fs+role","fs+role+sc","bag+role+zcot", "role+zcot+sc","algo+bag+role", "bag+role+sc", "algo+role+sc","fs+bag+sc", "algo+bag+sc", "zcot+bag+sc", 
              "bag+fs+role+sc", "bag+role+zcot+sc", "algo+bag+role+sc","bag+fs+role+zcot", "algo+bag+fs+role", "fs+role+zcot+sc", "algo+fs+role+sc", "fs+algo+bag+sc", "fs+bag+zcot+sc",
              "bag+fs+role+zcot+sc", "algo+bag+fs+role+sc"]
promptinglist_for_combinations = ["fs", "zcot", "fs+zcot", "bag+fs", "bag+fs+zcot"]
promptings_for_combinations = {a: {b: "" for b in filteredForQuestionID} for a in promptinglist_for_combinations}

bagSentence = "Let's construct a graph with the nodes and edges first. "
stepByStepSentence = "Let's think step by step. "
ex1_desc= "A directed graph contains the nodes: 1,2,3. There is an edge from 1 to 2 with weight 5. There is an edge from 1 to 3 with weight 7. There is an edge from 2 to 3 with weight 3. "
ex2_desc= "A directed graph contains the nodes: P,Q,R,S. There is an edge from P to Q with weight 5. There is an edge from Q to R with weight 7. There is an edge from R to S with weight 2. There is an edge from S to P with weight 3. There is an edge from P to R with weight 4. "
yes_answer = "Answer: Yes. "
no_answer = "Answer: No. "
roles = {
    "Directly follow - Y-N": "As a graph analysis expert, your task is to determine whether there is a direct follower of another node in a directed graph. ",
    "Direct successor - Y-N": "As a graph analysis expert, your task is to determine whether there is a direct successor of another node in a directed graph. ",
    "Directly reachable nodes - Number": "As a graph analysis expert, your task is to determine how many nodes are directly reachable from a node in a directed graph. ",
    "Reachable nodes - Number": "As a graph analysis expert, your task is to determine how many nodes are reachable from a node in a directed graph. ",
    "No successors": "As a graph analysis expert, your task is to determine all nodes without a successor in a directed graph. ",
    "Direct predecessor - Y-N": "As a graph analysis expert, your task is to determine whether a node is a direct successor of another node in a directed graph. ",
    "Directly reachable from nodes - Number": "As a graph analysis expert, your task is to determine from how many nodes a specific node is directly reachable in a directed graph. ",
    "Reachable from nodes - Number": "As a graph analysis expert, your task is to determine from how many nodes a specific node is reachable in a directed graph. ",
    "No predecessors": "As a graph analysis expert, your task is to determine all nodes without a predecessor in a directed graph. ",
    "Node existence - Y-N": "As a graph analysis expert, your task is to determine if a node exists in a directed graph. ",
    "Number of nodes - Number": "As a graph analysis expert, your task is to determine how many nodes exist in a directed graph. ",
    "Number of edges - Number": "As a graph analysis expert, your task is to determine how many edges exists in a directed graph. ",
    "Highest degree node - Node": "As a graph analysis expert, your task is to determine which nodes have the highest degree in a directed graph. ",
    "Highest degree - Number": "As a graph analysis expert, your task is to determine what the highest degree of all nodes in a directed graph is. ",
    "Cycle existence - Y-N": "As a graph analysis expert, your task is to determine whether there exists a cycle in a directed graph. ",
    "Number of Cycles - Number": "As a graph analysis expert, your task is to determine how many simple cycles are contained in a directed graph. ",
    "Concurrency Check - Y-N": "As a graph analysis expert, your task is to determine whether two nodes are concurrent in a directed graph. ",
    "Hamilton path existence - Y-N": "As a graph analysis expert, your task is to determine whether there exists a Hamilton path in a directed graph. ",
    "Length of shortest path - Number": "As a graph analysis expert, your task is to determine the length of the shortest simple path between two nodes in a directed graph. ",
    "Shortest path - Path": "As a graph analysis expert, your task is to determine the shortest simple path between two nodes in a directed graph. ",
    "Length of longest path - Number": "As a graph analysis expert, your task is to determine the length of the longest simple path between two nodes in a directed graph. ",
    "Longest path - Path": "As a graph analysis expert, your task is to determine the longest simple path between two nodes in a directed graph. ",
    "Path existence - Y-N": "As a graph analysis expert, your task is to determine whether there is a path between two nodes in a directed graph. ",
    "Path variants count - Number":""
}
algorithms = {
    "Directly follow - Y-N": "Check if there is a directed edge from the source node to the target node in the graph's adjacency list or matrix. ",
    "Direct successor - Y-N": "Check if the target node appears in the adjacency list of the source node (i.e., if there is a direct edge). ",
    "Directly reachable nodes - Number": "Return the length of the adjacency list of the given node, i.e., count all direct neighbors. ",
    "Reachable nodes - Number": "Use a Depth-First Search (DFS) or Breadth-First Search (BFS) to explore all nodes reachable from the source node and count them. ",
    "No successors": "Iterate through all nodes and return those with an empty adjacency list (i.e., no outgoing edges). ",
    "Direct predecessor - Y-N": "Check if the source node appears in the adjacency list of the target node (reverse edge direction). ",
    "Directly reachable from nodes - Number": "Count how many adjacency lists contain the target node (i.e., how many direct incoming edges it has). ",
    "Reachable from nodes - Number": "Run a DFS or BFS from every node and count how many can reach the target node. ",
    "No predecessors": "Iterate through all nodes and identify those not appearing in any adjacency list (i.e., zero incoming edges). ",
    "Node existence - Y-N": "Check if the node is in the list or set of nodes in the graph. ",
    "Number of nodes - Number": "Return the number of unique nodes in the graph representation. ",
    "Number of edges - Number": "Count the total number of entries across all adjacency lists (or count all 1s in the adjacency matrix). ",
    "Highest degree node - Node": "Calculate in-degree + out-degree for each node and return node(s) with the highest total. ",
    "Highest degree - Number": "Calculate in-degree + out-degree for each node and return the maximum value. ",
    "Cycle existence - Y-N": "Use DFS with a recursion stack to detect back edges, which indicate cycles in a directed graph. ",
    "Number of Cycles - Number": "Use Johnson's algorithm to enumerate all simple cycles and count them. ",
    "Concurrency Check - Y-N": "Check if two nodes are mutually unreachable using two DFS traversals (one from each node). ",
    "Hamilton path existence - Y-N": "Use backtracking or dynamic programming (e.g., Held-Karp algorithm) to test for the existence of a path visiting each node exactly once. ",
    "Length of shortest path - Number": "Use BFS (unweighted graph) or Dijkstra's algorithm (weighted graph) to find the shortest path and return its length. ",
    "Shortest path - Path": "Use BFS or Dijkstra's algorithm and reconstruct the shortest path from the predecessor map. ",
    "Length of longest path - Number": "In a Directed Acyclic Graph (DAG), use topological sorting followed by dynamic programming to find the longest path length. ",
    "Longest path - Path": "In a Directed Acyclic Graph (DAG), use topological sorting followed by dynamic programming to find the longest path and reconstruct the actual path along with the length. ",
    "Path existence - Y-N": "We can use a Depth-First Search (DFS) algorithm to determine if there is a path between two nodes in a directed graph. In DFS, we start at the source node and explore as far as possible along each branch before backtracking. ",
    "Path variants count - Number": "Use DFS and count all simple paths from source to target without revisiting nodes (exponential in general case). "
}

promptings_for_combinations['fs']['Number of edges - Number'] = ex1_desc + "Question: How many edges contains the graph? " + "Answer: 3. " + ex2_desc + "Question: How many edges contains the graph? " + "Answer: 5. " 
promptings_for_combinations['fs']['Number of nodes - Number'] = ex1_desc + "Question: How many nodes contains the graph? " + "Answer: 3. " + ex2_desc + "Question: How many nodes contains the graph?  " + "Answer: 4. " 
promptings_for_combinations['fs']['Node existence - Y-N'] = ex1_desc + "Question: Exists a node called 3? " + yes_answer + ex2_desc + "Question: Exists a node called T? " + no_answer
promptings_for_combinations['fs']['Highest degree node - Node'] = ex1_desc + "Question: What are the names of the nodes with the highest degree?  " + "Answer: 1,2,3. " + ex2_desc + "Question: What are the names of the nodes with the highest degree?  " + "Answer: P,R. "
promptings_for_combinations['fs']['Highest degree - Number'] = ex1_desc + "Question: What is the highest degree of all nodes in the graph? " + "Answer: 2. " + ex2_desc + "Question: What is the highest degree of all nodes in the graph? " + "Answer: 3. "
promptings_for_combinations['fs']['Directly follow - Y-N'] = ex1_desc + "Question: Does node 3 directly follow node 1? " + yes_answer + ex2_desc + "Question: Does node P directly follow node R? " + no_answer
promptings_for_combinations['fs']['Directly reachable from nodes - Number'] =  ex1_desc + "Question: From how many nodes is node 2 directly reachable? " + "Answer: 1. " + ex2_desc + "Question: From how many nodes is node R directly reachable? " + "Answer: 2. "
promptings_for_combinations['fs']['Direct predecessor - Y-N'] = ex1_desc + "Question: Is node 1 a direct predecessor of node 3? " + yes_answer + ex2_desc + "Question: Is node S a direct predecessor of node Q? " + no_answer
promptings_for_combinations['fs']['Reachable nodes - Number'] = ex1_desc + "Question: How many nodes are reachable from node 2? " + "Answer: 1. " + ex2_desc + "Question: How many nodes are reachable from node P? " + "Answer: 3. "
promptings_for_combinations['fs']['No successors'] = ex1_desc + "Question: Which nodes have no successors? " + "Answer: 3. " + ex2_desc + "Question: Which nodes have no successors? " + "Answer: None. "
promptings_for_combinations['fs']['Direct successor - Y-N'] = ex1_desc + "Question: Is node 2 a direct successor of node 1? " + yes_answer + ex2_desc + "Question: Is node S a direct successor of node Q? " + no_answer
promptings_for_combinations['fs']['Directly reachable nodes - Number'] = ex1_desc + "Question: How many nodes can be reached directly from node 1? " + "Answer: 2. " + ex2_desc + "Question: How many nodes can be reached directly from node P? " + "Answer: 2. "
promptings_for_combinations['fs']['Reachable from nodes - Number'] =ex1_desc + "Question: From how many nodes is node 1 reachable? " + "Answer: 0. " + ex2_desc + "Question: From how many nodes is node Q reachable? " + "Answer: 3. "
promptings_for_combinations['fs']['No predecessors'] = ex1_desc + "Question: Which nodes have no predecessor? " + "Answer: 1. "  + ex2_desc + "Question: Which nodes have no predecessor? " + "Answer: None. " 
promptings_for_combinations['fs']['Cycle existence - Y-N'] = ex1_desc + "Question: Does the graph contain any cycles? " + no_answer + ex2_desc + "Question: Does the graph contain any cycles? " + yes_answer
promptings_for_combinations['fs']['Number of Cycles - Number'] =ex1_desc + "Question: How many simple cycles contains the graph? " + "Answer: 0. " + ex2_desc + "Question: How many simple cycles contains the graph? " + "Answer: 7. "
promptings_for_combinations['fs']['Concurrency Check - Y-N'] = ex1_desc + "Question: Are node 2 and 3 concurrent? " + no_answer + ex2_desc + "Question: Are node Q and S concurrent? " + no_answer
promptings_for_combinations['fs']['Hamilton path existence - Y-N'] = ex1_desc + "Question: Is there a Hamilton path in the graph? " + yes_answer + ex2_desc + "Question: Is there a Hamilton path in the graph? " + yes_answer
promptings_for_combinations['fs']["Length of shortest path - Number"] =ex1_desc + "Question: What is the length of the shortest simple path from node 1 to node 3? " + "Answer: 7. " + ex2_desc + "Question: What is the length of the shortest simple path from node P to node S? " + "Answer: 6. "
promptings_for_combinations['fs']['Shortest path - Path'] =ex1_desc + "Question: What is the shortest simple path from node 1 to node 3? " + "Answer: 1,3. " + ex2_desc + "Question: What is the shortest simple path from node P to node S? " + "Answer: P,R,S. "
promptings_for_combinations['fs']['Length of longest path - Number'] = ex1_desc + "Question: What is the length of the longest simple path from node 1 to node 3? " + "Answer: 8. " + ex2_desc + "Question: What is the length of the longest simple path from node P to node S? " + "Answer: 14. "
promptings_for_combinations['fs']['Path existence - Y-N'] = ex1_desc + "Question: Is there a path from node 1 to node 3? " + yes_answer + ex2_desc + "Question: Is there a path from node 2 to node 1? " + no_answer
promptings_for_combinations['fs']['Longest path - Path'] = ex1_desc + "Question: What is the longest simple path from node 1 to node 3? " + "Answer: 1,2,3. " + ex2_desc + "Question: What is the longest simple path from node P to node S? " + "Answer: P,Q,R,S. "


promptings_for_combinations['bag+fs']['Number of edges - Number'] = ex1_desc + bagSentence + "Question: How many edges contains the graph? " + "Answer: 3. " + ex2_desc + bagSentence + "Question: How many edges contains the graph? " + "Answer: 5. " 
promptings_for_combinations['bag+fs']['Number of nodes - Number'] = ex1_desc + bagSentence + "Question: How many nodes contains the graph? " + "Answer: 3. " + ex2_desc + bagSentence + "Question: How many nodes contains the graph?  " + "Answer: 4. " 
promptings_for_combinations['bag+fs']['Node existence - Y-N'] = ex1_desc + bagSentence + "Question: Exists a node called 3? " + yes_answer + ex2_desc + bagSentence + "Question: Exists a node called T? " + no_answer
promptings_for_combinations['bag+fs']['Highest degree node - Node'] = ex1_desc + bagSentence + "Question: What are the names of the nodes with the highest degree?  " + "Answer: 1,2,3. " + ex2_desc + bagSentence + "Question: What are the names of the nodes with the highest degree?  " + "Answer: P,R. "
promptings_for_combinations['bag+fs']['Highest degree - Number'] = ex1_desc + bagSentence + "Question: What is the highest degree of all nodes in the graph? " + "Answer: 2. " + ex2_desc + bagSentence + "Question: What is the highest degree of all nodes in the graph? " + "Answer: 3. "
promptings_for_combinations['bag+fs']['Directly follow - Y-N'] = ex1_desc + bagSentence + "Question: Does node 3 directly follow node 1? " + yes_answer + ex2_desc + bagSentence + "Question: Does node P directly follow node R? " + no_answer
promptings_for_combinations['bag+fs']['Directly reachable from nodes - Number'] =  ex1_desc + bagSentence + "Question: From how many nodes is node 2 directly reachable? " + "Answer: 1. " + ex2_desc + bagSentence + "Question: From how many nodes is node R directly reachable? " + "Answer: 2. "
promptings_for_combinations['bag+fs']['Direct predecessor - Y-N'] = ex1_desc + bagSentence + "Question: Is node 1 a direct predecessor of node 3? " + yes_answer + ex2_desc + bagSentence + "Question: Is node S a direct predecessor of node Q? " + no_answer
promptings_for_combinations['bag+fs']['Reachable nodes - Number'] = ex1_desc + bagSentence + "Question: How many nodes are reachable from node 2? " + "Answer: 1. " + ex2_desc + bagSentence + "Question: How many nodes are reachable from node P? " + "Answer: 3. "
promptings_for_combinations['bag+fs']['No successors'] = ex1_desc + bagSentence + "Question: Which nodes have no successors? " + "Answer: 3. " + ex2_desc + bagSentence + "Question: Which nodes have no successors? " + "Answer: None. "
promptings_for_combinations['bag+fs']['Direct successor - Y-N'] = ex1_desc + bagSentence + "Question: Is node 2 a direct successor of node 1? " + yes_answer + ex2_desc + bagSentence + "Question: Is node S a direct successor of node Q? " + no_answer
promptings_for_combinations['bag+fs']['Directly reachable nodes - Number'] = ex1_desc + bagSentence + "Question: How many nodes can be reached directly from node 1? " + "Answer: 2. " + ex2_desc + bagSentence + "Question: How many nodes can be reached directly from node P? " + "Answer: 2. "
promptings_for_combinations['bag+fs']['Reachable from nodes - Number'] =ex1_desc + bagSentence + "Question: From how many nodes is node 1 reachable? " + "Answer: 0. " + ex2_desc + bagSentence + "Question: From how many nodes is node Q reachable? " + "Answer: 3. "
promptings_for_combinations['bag+fs']['No predecessors'] = ex1_desc + bagSentence + "Question: Which nodes have no predecessor? " + "Answer: 1. "  + ex2_desc + bagSentence + "Question: Which nodes have no predecessor? " + "Answer: None. " 
promptings_for_combinations['bag+fs']['Cycle existence - Y-N'] = ex1_desc + bagSentence + "Question: Does the graph contain any cycles? " + no_answer + ex2_desc + bagSentence + "Question: Does the graph contain any cycles? " + yes_answer
promptings_for_combinations['bag+fs']['Number of Cycles - Number'] =ex1_desc + bagSentence + "Question: How many simple cycles contains the graph? " + "Answer: 0. " + ex2_desc + bagSentence + "Question: How many simple cycles contains the graph? " + "Answer: 7. "
promptings_for_combinations['bag+fs']['Concurrency Check - Y-N'] = ex1_desc + bagSentence + "Question: Are node 2 and 3 concurrent? " + no_answer + ex2_desc + bagSentence + "Question: Are node Q and S concurrent? " + no_answer
promptings_for_combinations['bag+fs']['Hamilton path existence - Y-N'] = ex1_desc + bagSentence + "Question: Is there a Hamilton path in the graph? " + yes_answer + ex2_desc + bagSentence + "Question: Is there a Hamilton path in the graph? " + yes_answer
promptings_for_combinations['bag+fs']["Length of shortest path - Number"] =ex1_desc + bagSentence + "Question: What is the length of the shortest simple path from node 1 to node 3? " + "Answer: 7. " + ex2_desc + bagSentence + "Question: What is the length of the shortest simple path from node P to node S? " + "Answer: 6. "
promptings_for_combinations['bag+fs']['Shortest path - Path'] =ex1_desc + bagSentence + "Question: What is the shortest simple path from node 1 to node 3? " + "Answer: 1,3. " + ex2_desc + bagSentence + "Question: What is the shortest simple path from node P to node S? " + "Answer: P,R,S. "
promptings_for_combinations['bag+fs']['Length of longest path - Number'] = ex1_desc + bagSentence + "Question: What is the length of the longest simple path from node 1 to node 3? " + "Answer: 8. " + ex2_desc + bagSentence + "Question: What is the length of the longest simple path from node P to node S? " + "Answer: 14. "
promptings_for_combinations['bag+fs']['Path existence - Y-N'] = ex1_desc + bagSentence + "Question: Is there a path from node 1 to node 3? " + yes_answer + ex2_desc + bagSentence + "Question: Is there a path from node 2 to node 1? " + no_answer
promptings_for_combinations['bag+fs']['Longest path - Path'] = ex1_desc + bagSentence + "Question: What is the longest simple path from node 1 to node 3? " + "Answer: 1,2,3. " + ex2_desc + bagSentence + "Question: What is the longest simple path from node P to node S? " + "Answer: P,Q,R,S. "


promptings_for_combinations['zcot']['Number of edges - Number'] = "Count all direct edges between nodes by traversing the graph and incrementing a counter for each found edge. "
promptings_for_combinations['zcot']['Number of nodes - Number'] = "Extract all unique node identifiers from the graph structure and count them to determine the total number of nodes. "
promptings_for_combinations['zcot']['Node existence - Y-N'] = "Check if the given node identifier is present in the set of all nodes of the graph. "
promptings_for_combinations['zcot']['Highest degree node - Node'] = "Calculate the sum of in-degree and out-degree for each node. Return the node(s) with the highest total degree. "
promptings_for_combinations['zcot']['Highest degree - Number'] = "Compute the degree (in-degree + out-degree) of all nodes and return the maximum degree value. "
promptings_for_combinations['zcot']['Directly follow - Y-N'] = "Inspect the adjacency list or matrix to see if there is a direct edge from the first node to the second. "
promptings_for_combinations['zcot']['Directly reachable nodes - Number'] = "Count how many outgoing neighbors are listed for a node in the adjacency list to determine direct reachability. "
promptings_for_combinations['zcot']['Direct successor - Y-N'] = "Check if the second node appears as a neighbor in the adjacency list of the first node. "
promptings_for_combinations['zcot']['Reachable nodes - Number'] = "Perform a traversal (DFS or BFS) from the start node and count how many nodes are reached. "
promptings_for_combinations['zcot']['No successors'] = "Identify all nodes that have no outgoing edges by checking for empty adjacency lists. "
promptings_for_combinations['zcot']['Direct predecessor - Y-N'] = "Check all adjacency lists to see if the first node appears as a neighbor of the second node. "
promptings_for_combinations['zcot']['Directly reachable from nodes - Number'] = "Count how many nodes have a direct edge pointing to the given node. "
promptings_for_combinations['zcot']['Reachable from nodes - Number'] = "Perform reverse reachability checks: for each node, test whether it can reach the given target and count successful cases. "
promptings_for_combinations['zcot']['No predecessors'] = "Return all nodes that are not found in any adjacency list as a neighbor (i.e., no incoming edges). "
promptings_for_combinations['zcot']['Cycle existence - Y-N'] = "Perform a DFS with a recursion stack to check if any node is revisited during the traversal, indicating a cycle. "
promptings_for_combinations['zcot']['Number of Cycles - Number'] = "Use a cycle enumeration algorithm (e.g., Johnson's) to detect and count all simple cycles in the graph. "
promptings_for_combinations['zcot']['Concurrency Check - Y-N'] = "Check if both nodes are not reachable from each other using separate reachability checks from each node. "
promptings_for_combinations['zcot']['Hamilton path existence - Y-N'] = "Try all permutations of the node set and test whether each consecutive node pair has a direct edge, or use backtracking with pruning. "
promptings_for_combinations['zcot']["Length of shortest path - Number"] = "Use BFS (unweighted) or Dijkstra's algorithm (weighted) to find the shortest path and return the number of steps or total weight. "
promptings_for_combinations['zcot']['Shortest path - Path'] = "Use a shortest path algorithm (e.g., BFS or Dijkstra) and reconstruct the actual sequence of nodes along the shortest path. "
promptings_for_combinations['zcot']['Length of longest path - Number'] = "If the graph is a DAG, use topological sort and dynamic programming to compute the longest simple path length. "
promptings_for_combinations['zcot']['Longest path - Path'] = "In a DAG, apply dynamic programming over a topological ordering to reconstruct the longest path. "
promptings_for_combinations['zcot']['Path existence - Y-N'] = "To determine whether a path exists, check all reachable nodes from the start node. Repeat this until the end node is in the reachable nodes set or the reachable nodes set does not change after an iteration. "


promptings_for_combinations['fs+zcot']['Number of edges - Number'] = stepByStepSentence + ex1_desc +promptings_for_combinations['zcot']['Number of edges - Number'] + "Question: How many edges contains the graph? " +"The graph has an edge from 1 to 2. Another edge from 1 to 3. And another one from 2 to 3. "+ "Answer: 3. " + ex2_desc +promptings_for_combinations['zcot']['Number of edges - Number'] + "Question: How many edges contains the graph? " +"There is an edge from P to Q. Another one fromQ to R. Another one from R to P. Another one from R to S. And one from S to P. "+ "Answer: 5. " 
promptings_for_combinations['fs+zcot']['Number of nodes - Number'] = stepByStepSentence + ex1_desc +promptings_for_combinations['zcot']['Number of nodes - Number'] + "Question: How many nodes contains the graph? "+"The graph has the nodes 1,2,3. " + "Answer: 3. " + ex2_desc+promptings_for_combinations['zcot']['Number of nodes - Number'] + "Question: How many nodes contains the graph?  " +"The graph has the nodes P,Q,R,S. "+ "Answer: 4. " 
promptings_for_combinations['fs+zcot']['Node existence - Y-N'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Node existence - Y-N'] + "Question: Exists a node called 3? " +"The graph has the nodes 1,2,3. This list contains a node called 3. "+ yes_answer +  ex2_desc+promptings_for_combinations['zcot']['Node existence - Y-N'] + "Question: Exists a node called T? "+"The graph contains the nodes P,Q,R,S. There is no node called T in thid node list. " + no_answer
promptings_for_combinations['fs+zcot']['Highest degree node - Node'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Highest degree node - Node'] + "Question: What are the names of the nodes with the highest degree?  "+"Node 1 has degree 2. Node 2 has degree 2. Node 3 has degree 2. So the highest degree is 2. " + "Answer: 1,2,3. " + ex2_desc+promptings_for_combinations['zcot']['Highest degree node - Node'] + "Question: What are the names of the nodes with the highest degree?  "+"Node P has degree 3. Node Q has degree 2. Node R has degree 3. Node S has degree 2. So the highest degree is 3." + "Answer: P,R. "
promptings_for_combinations['fs+zcot']['Highest degree - Number'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Highest degree - Number'] + "Question: What is the highest degree of all nodes in the graph? "+"Node 1 has degree 2. Node 2 has degree 2. Node 3 has degree 2. " + "Answer: 2. " + ex2_desc +promptings_for_combinations['zcot']['Highest degree - Number']+ "Question: What is the highest degree of all nodes in the graph? " +"Node P has degree 3. Node Q has degree 2. Node R has degree 3. Node S has degree 2. "+ "Answer: 3. "
promptings_for_combinations['fs+zcot']['Directly follow - Y-N'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Directly follow - Y-N'] + "Question: Does node 3 directly follow node 1? " +"The graph contains an edge from 1 to 3. "+ yes_answer + ex2_desc +promptings_for_combinations['zcot']['Directly follow - Y-N']+ "Question: Does node P directly follow node R? "+"The graph does not contain an edge from R to P. " + no_answer
promptings_for_combinations['fs+zcot']['Directly reachable from nodes - Number'] =  stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Directly reachable from nodes - Number'] + "Question: From how many nodes is node 2 directly reachable? "+"The only edge with node 2 as endpoint is the edge from node 1 to node 2.  " + "Answer: 1. " + ex2_desc+promptings_for_combinations['zcot']['Directly reachable from nodes - Number'] + "Question: From how many nodes is node R directly reachable? "+"The only edges with node R as endpoint are the edge from node P to R and from node Q to R. " + "Answer: 2. "
promptings_for_combinations['fs+zcot']['Direct predecessor - Y-N'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Direct predecessor - Y-N'] + "Question: Is node 1 a direct predecessor of node 3? " +"The graph contains an edge from 1 to 3. "+ yes_answer + ex2_desc+promptings_for_combinations['zcot']['Direct predecessor - Y-N'] + "Question: Is node S a direct predecessor of node Q? "+"The graph does not contain an edge from node S to node Q. " + no_answer
promptings_for_combinations['fs+zcot']['Reachable nodes - Number'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Reachable nodes - Number'] + "Question: How many nodes are reachable from node 2? " +"Node 2 has only one outgoing edge to node 3. Node 3 has no outgoing edge. "+ "Answer: 1. " + ex2_desc +promptings_for_combinations['zcot']['Reachable nodes - Number']+ "Question: How many nodes are reachable from node P? "+"P has outgoing edges to the nodes Q and R. R has an outgoing edge to S. ALl nodes from the graph are reachable from P. " + "Answer: 3. "
promptings_for_combinations['fs+zcot']['No successors'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['No successors'] + "Question: Which nodes have no successors? "+"Node 3 has no outgoing edge. "+ "Answer: 3. " + ex2_desc+promptings_for_combinations['zcot']['No successors'] + "Question: Which nodes have no successors? "+"All nodes of the graph have at least one outgoing edge. " + "Answer: None. "
promptings_for_combinations['fs+zcot']['Direct successor - Y-N'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Direct successor - Y-N'] + "Question: Is node 2 a direct successor of node 1? "+"The graph contains an edge from node 1 to node 2. " + yes_answer + ex2_desc+promptings_for_combinations['zcot']['Direct successor - Y-N'] + "Question: Is node S a direct successor of node Q? " +"The graph contains no node from node Q to node S. "+ no_answer
promptings_for_combinations['fs+zcot']['Directly reachable nodes - Number'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Directly reachable nodes - Number'] + "Question: How many nodes can be reached directly from node 1? "+"Node 1 has an outgoing edge to node 2 and to node 3. " + "Answer: 2. " + ex2_desc+promptings_for_combinations['zcot']['Directly reachable nodes - Number'] + "Question: How many nodes can be reached directly from node P? "+"Node P has an outgoing edge to node Q and to node R. " + "Answer: 2. "
promptings_for_combinations['fs+zcot']['Reachable from nodes - Number'] =stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Reachable from nodes - Number'] + "Question: From how many nodes is node 1 reachable? "+"Node 1 has no incoming edges. " + "Answer: 0. " + ex2_desc+promptings_for_combinations['zcot']['Reachable from nodes - Number'] + "Question: From how many nodes is node Q reachable? "+"Node Q haa an incoming edge from node P. Node P has an incoming edge from S. S has an incoming edge from R. So all nodes can reach node Q. " + "Answer: 3. "
promptings_for_combinations['fs+zcot']['No predecessors'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['No predecessors'] + "Question: Which nodes have no predecessor? "+"Only node 3 has no outgoing edges. " + "Answer: 1. "  + ex2_desc+promptings_for_combinations['zcot']['No predecessors']  + "Question: Which nodes have no predecessor? "+"All nodes have at least one outgoing edge. " + "Answer: None. " 
promptings_for_combinations['fs+zcot']['Cycle existence - Y-N'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Cycle existence - Y-N'] + "Question: Does the graph contain any cycles? " +"There is no path from node 1 to node 1. There is no path from node 2 to node 2. There is no path from node 3 to node 3. "+ no_answer + ex2_desc+promptings_for_combinations['zcot']['Cycle existence - Y-N']  + "Question: Does the graph contain any cycles? " +"P has an outgoing edge to R. R has an outgoing edge to S. S ha an outgoing edge to P. So there is a path from node P to node P. "+ yes_answer
promptings_for_combinations['fs+zcot']['Number of Cycles - Number'] =stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Number of Cycles - Number']  + "Question: How many simple cycles contains the graph? " +"There is no path from node 1 to node 1. There is no path from node 2 to node 2. There is no path from node 3 to node 3. "+ "Answer: 0. " + ex2_desc +promptings_for_combinations['zcot']['Number of Cycles - Number']+ "Question: How many simple cycles contains the graph? "+"There are two simple paths from node P to P. There is one path from node Q to Q. There are two simple paths from R to R. There are two simple paths from node S to S. " + "Answer: 7. "
promptings_for_combinations['fs+zcot']['Concurrency Check - Y-N'] = stepByStepSentence + ex1_desc +promptings_for_combinations['zcot']['Concurrency Check - Y-N'] + "Question: Are node 2 and 3 concurrent? " +"There is a path from node 2 to 3. There is no path from node 3 to 2. "+ no_answer + ex2_desc+promptings_for_combinations['zcot']['Concurrency Check - Y-N'] + "Question: Are node Q and S concurrent? " +"There is a path from node Q to S. There is a path from node S to Q. "+ no_answer
promptings_for_combinations['fs+zcot']['Hamilton path existence - Y-N'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Hamilton path existence - Y-N']  + "Question: Is there a Hamilton path in the graph? " +"The graph contains the edge from node 1 to 2. And an edge from node 2 to 3. This path contains all nodes from the graph. "+ yes_answer + ex2_desc+promptings_for_combinations['zcot']['Hamilton path existence - Y-N']  + "Question: Is there a Hamilton path in the graph? "+"The graph contains an edge from P to Q. And from Q to R. And from R to S. This path contains all nodes of the graph. "+ yes_answer
promptings_for_combinations['fs+zcot']["Length of shortest path - Number"] =stepByStepSentence + ex1_desc +promptings_for_combinations['zcot']["Length of shortest path - Number"] + "Question: What is the length of the shortest simple path from node 1 to node 3? "+"There are two simple paths from node 1 to 3. The path 1,2,3 has the length 8. The path 1,3 has the length 7. " + "Answer: 7. " + ex2_desc+promptings_for_combinations['zcot']["Length of shortest path - Number"] + "Question: What is the length of the shortest simple path from node P to node S? "+"There are two simple paths from node P to S. The path P,Q,R,S has the length 14. The path P,R,S has the length 6. " + "Answer: 6. "
promptings_for_combinations['fs+zcot']['Shortest path - Path'] =stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Shortest path - Path'] + "Question: What is the shortest simple path from node 1 to node 3? " +"There are two simple paths from node 1 to 3. The path 1,2,3 has the length 8. The path 1,3 has the length 7. "+ "Answer: 1,3. " + ex2_desc+promptings_for_combinations['zcot']['Shortest path - Path'] + "Question: What is the shortest simple path from node P to node S? "+"There are two simple paths from node P to S. The path P,Q,R,S has the length 14. The path P,R,S has the length 6. " + "Answer: P,R,S. "
promptings_for_combinations['fs+zcot']['Length of longest path - Number'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Length of longest path - Number']  + "Question: What is the length of the longest simple path from node 1 to node 3? " +"There are two simple paths from node 1 to 3. The path 1,2,3 has the length 8. The path 1,3 has the length 7. "+ "Answer: 8. " + ex2_desc+promptings_for_combinations['zcot']['Length of longest path - Number'] + "Question: What is the length of the longest simple path from node P to node S? "+"There are two simple paths from node P to S. The path P,Q,R,S has the length 14. The path P,R,S has the length 6. " + "Answer: 14. "
promptings_for_combinations['fs+zcot']['Path existence - Y-N'] = stepByStepSentence + ex1_desc +promptings_for_combinations['zcot']['Path existence - Y-N'] + "Question: Is there a path from node 1 to node 3? "+ "There is an edge from node 1 to node 3. So there is a path from node 1 to node 3. "+ yes_answer + ex2_desc + promptings_for_combinations['zcot']['Path existence - Y-N']+ "Question: Is there a path from node 2 to node 1? "+"Node 2 has only one outgoing edge to node 3. Node 3 has no outgoing edge. So there is no path from node 2 to node 1. " + no_answer
promptings_for_combinations['fs+zcot']['Longest path - Path'] = stepByStepSentence + ex1_desc+promptings_for_combinations['zcot']['Longest path - Path'] + "Question: What is the longest simple path from node 1 to node 3? "+"There are two simple paths from node 1 to 3. The path 1,2,3 has the length 8. The path 1,3 has the length 7. " + "Answer: 1,2,3. " + ex2_desc+promptings_for_combinations['zcot']['Longest path - Path'] + "Question: What is the longest simple path from node P to node S? " +"There are two simple paths from node P to S. The path P,Q,R,S has the length 14. The path P,R,S has the length 6. "+ "Answer: P,Q,R,S. "


promptings_for_combinations['bag+fs+zcot']['Number of edges - Number'] = stepByStepSentence + ex1_desc+ bagSentence +promptings_for_combinations['zcot']['Number of edges - Number'] + "Question: How many edges contains the graph? " +"The graph has an edge from 1 to 2. Another edge from 1 to 3. And another one from 2 to 3. "+ "Answer: 3. " + ex2_desc + bagSentence+promptings_for_combinations['zcot']['Number of edges - Number'] + "Question: How many edges contains the graph? " +"There is an edge from P to Q. Another one fromQ to R. Another one from R to P. Another one from R to S. And one from S to P. "+ "Answer: 5. " 
promptings_for_combinations['bag+fs+zcot']['Number of nodes - Number'] = stepByStepSentence + ex1_desc + bagSentence+promptings_for_combinations['zcot']['Number of nodes - Number'] + "Question: How many nodes contains the graph? "+"The graph has the nodes 1,2,3. " + "Answer: 3. " + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Number of nodes - Number'] + "Question: How many nodes contains the graph?  " +"The graph has the nodes P,Q,R,S. "+ "Answer: 4. " 
promptings_for_combinations['bag+fs+zcot']['Node existence - Y-N'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Node existence - Y-N'] + "Question: Exists a node called 3? " +"The graph has the nodes 1,2,3. This list contains a node called 3. "+ yes_answer +  ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Node existence - Y-N'] + "Question: Exists a node called T? "+"The graph contains the nodes P,Q,R,S. There is no node called T in thid node list. " + no_answer
promptings_for_combinations['bag+fs+zcot']['Highest degree node - Node'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Highest degree node - Node'] + "Question: What are the names of the nodes with the highest degree?  "+"Node 1 has degree 2. Node 2 has degree 2. Node 3 has degree 2. So the highest degree is 2. " + "Answer: 1,2,3. " + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Highest degree node - Node'] + "Question: What are the names of the nodes with the highest degree?  "+"Node P has degree 3. Node Q has degree 2. Node R has degree 3. Node S has degree 2. So the highest degree is 3." + "Answer: P,R. "
promptings_for_combinations['bag+fs+zcot']['Highest degree - Number'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Highest degree - Number'] + "Question: What is the highest degree of all nodes in the graph? "+"Node 1 has degree 2. Node 2 has degree 2. Node 3 has degree 2. " + "Answer: 2. " + ex2_desc+ bagSentence +promptings_for_combinations['zcot']['Highest degree - Number']+ "Question: What is the highest degree of all nodes in the graph? " +"Node P has degree 3. Node Q has degree 2. Node R has degree 3. Node S has degree 2. "+ "Answer: 3. "
promptings_for_combinations['bag+fs+zcot']['Directly follow - Y-N'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Directly follow - Y-N'] + "Question: Does node 3 directly follow node 1? " +"The graph contains an edge from 1 to 3. "+ yes_answer + ex2_desc + bagSentence+promptings_for_combinations['zcot']['Directly follow - Y-N']+ "Question: Does node P directly follow node R? "+"The graph does not contain an edge from R to P. " + no_answer
promptings_for_combinations['bag+fs+zcot']['Directly reachable from nodes - Number'] =  stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Directly reachable from nodes - Number'] + "Question: From how many nodes is node 2 directly reachable? "+"The only edge with node 2 as endpoint is the edge from node 1 to node 2.  " + "Answer: 1. " + ex2_desc+promptings_for_combinations['zcot']['Directly reachable from nodes - Number'] + "Question: From how many nodes is node R directly reachable? "+"The only edges with node R as endpoint are the edge from node P to R and from node Q to R. " + "Answer: 2. "
promptings_for_combinations['bag+fs+zcot']['Direct predecessor - Y-N'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Direct predecessor - Y-N'] + "Question: Is node 1 a direct predecessor of node 3? " +"The graph contains an edge from 1 to 3. "+ yes_answer + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Direct predecessor - Y-N'] + "Question: Is node S a direct predecessor of node Q? "+"The graph does not contain an edge from node S to node Q. " + no_answer
promptings_for_combinations['bag+fs+zcot']['Reachable nodes - Number'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Reachable nodes - Number'] + "Question: How many nodes are reachable from node 2? " +"Node 2 has only one outgoing edge to node 3. Node 3 has no outgoing edge. "+ "Answer: 1. " + ex2_desc + bagSentence+promptings_for_combinations['zcot']['Reachable nodes - Number']+ "Question: How many nodes are reachable from node P? "+"P has outgoing edges to the nodes Q and R. R has an outgoing edge to S. ALl nodes from the graph are reachable from P. " + "Answer: 3. "
promptings_for_combinations['bag+fs+zcot']['No successors'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['No successors'] + "Question: Which nodes have no successors? "+"Node 3 has no outgoing edge. "+ "Answer: 3. " + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['No successors'] + "Question: Which nodes have no successors? "+"All nodes of the graph have at least one outgoing edge. " + "Answer: None. "
promptings_for_combinations['bag+fs+zcot']['Direct successor - Y-N'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Direct successor - Y-N'] + "Question: Is node 2 a direct successor of node 1? "+"The graph contains an edge from node 1 to node 2. " + yes_answer + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Direct successor - Y-N'] + "Question: Is node S a direct successor of node Q? " +"The graph contains no node from node Q to node S. "+ no_answer
promptings_for_combinations['bag+fs+zcot']['Directly reachable nodes - Number'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Directly reachable nodes - Number'] + "Question: How many nodes can be reached directly from node 1? "+"Node 1 has an outgoing edge to node 2 and to node 3. " + "Answer: 2. " + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Directly reachable nodes - Number'] + "Question: How many nodes can be reached directly from node P? "+"Node P has an outgoing edge to node Q and to node R. " + "Answer: 2. "
promptings_for_combinations['bag+fs+zcot']['Reachable from nodes - Number'] =stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Reachable from nodes - Number'] + "Question: From how many nodes is node 1 reachable? "+"Node 1 has no incoming edges. " + "Answer: 0. " + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Reachable from nodes - Number'] + "Question: From how many nodes is node Q reachable? "+"Node Q haa an incoming edge from node P. Node P has an incoming edge from S. S has an incoming edge from R. So all nodes can reach node Q. " + "Answer: 3. "
promptings_for_combinations['bag+fs+zcot']['No predecessors'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['No predecessors'] + "Question: Which nodes have no predecessor? "+"Only node 3 has no outgoing edges. " + "Answer: 1. "  + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['No predecessors']  + "Question: Which nodes have no predecessor? "+"All nodes have at least one outgoing edge. " + "Answer: None. " 
promptings_for_combinations['bag+fs+zcot']['Cycle existence - Y-N'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Cycle existence - Y-N'] + "Question: Does the graph contain any cycles? " +"There is no path from node 1 to node 1. There is no path from node 2 to node 2. There is no path from node 3 to node 3. "+ no_answer + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Cycle existence - Y-N']  + "Question: Does the graph contain any cycles? " +"P has an outgoing edge to R. R has an outgoing edge to S. S ha an outgoing edge to P. So there is a path from node P to node P. "+ yes_answer
promptings_for_combinations['bag+fs+zcot']['Number of Cycles - Number'] =stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Number of Cycles - Number']  + "Question: How many simple cycles contains the graph? " +"There is no path from node 1 to node 1. There is no path from node 2 to node 2. There is no path from node 3 to node 3. "+ "Answer: 0. " + ex2_desc+ bagSentence +promptings_for_combinations['zcot']['Number of Cycles - Number']+ "Question: How many simple cycles contains the graph? "+"There are two simple paths from node P to P. There is one path from node Q to Q. There are two simple paths from R to R. There are two simple paths from node S to S. " + "Answer: 7. "
promptings_for_combinations['bag+fs+zcot']['Concurrency Check - Y-N'] = stepByStepSentence + ex1_desc+ bagSentence +promptings_for_combinations['zcot']['Concurrency Check - Y-N'] + "Question: Are node 2 and 3 concurrent? " +"There is a path from node 2 to 3. There is no path from node 3 to 2. "+ no_answer + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Concurrency Check - Y-N'] + "Question: Are node Q and S concurrent? " +"There is a path from node Q to S. There is a path from node S to Q. "+ no_answer
promptings_for_combinations['bag+fs+zcot']['Hamilton path existence - Y-N'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Hamilton path existence - Y-N']  + "Question: Is there a Hamilton path in the graph? " +"The graph contains the edge from node 1 to 2. And an edge from node 2 to 3. This path contains all nodes from the graph. "+ yes_answer + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Hamilton path existence - Y-N']  + "Question: Is there a Hamilton path in the graph? "+"The graph contains an edge from P to Q. And from Q to R. And from R to S. This path contains all nodes of the graph. "+ yes_answer
promptings_for_combinations['bag+fs+zcot']["Length of shortest path - Number"] =stepByStepSentence + ex1_desc + bagSentence+promptings_for_combinations['zcot']["Length of shortest path - Number"] + "Question: What is the length of the shortest simple path from node 1 to node 3? "+"There are two simple paths from node 1 to 3. The path 1,2,3 has the length 8. The path 1,3 has the length 7. " + "Answer: 7. " + ex2_desc+ bagSentence+promptings_for_combinations['zcot']["Length of shortest path - Number"] + "Question: What is the length of the shortest simple path from node P to node S? "+"There are two simple paths from node P to S. The path P,Q,R,S has the length 14. The path P,R,S has the length 6. " + "Answer: 6. "
promptings_for_combinations['bag+fs+zcot']['Shortest path - Path'] =stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Shortest path - Path'] + "Question: What is the shortest simple path from node 1 to node 3? " +"There are two simple paths from node 1 to 3. The path 1,2,3 has the length 8. The path 1,3 has the length 7. "+ "Answer: 1,3. " + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Shortest path - Path'] + "Question: What is the shortest simple path from node P to node S? "+"There are two simple paths from node P to S. The path P,Q,R,S has the length 14. The path P,R,S has the length 6. " + "Answer: P,R,S. "
promptings_for_combinations['bag+fs+zcot']['Length of longest path - Number'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Length of longest path - Number']  + "Question: What is the length of the longest simple path from node 1 to node 3? " +"There are two simple paths from node 1 to 3. The path 1,2,3 has the length 8. The path 1,3 has the length 7. "+ "Answer: 8. " + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Length of longest path - Number'] + "Question: What is the length of the longest simple path from node P to node S? "+"There are two simple paths from node P to S. The path P,Q,R,S has the length 14. The path P,R,S has the length 6. " + "Answer: 14. "
promptings_for_combinations['bag+fs+zcot']['Path existence - Y-N'] = stepByStepSentence + ex1_desc+ bagSentence +promptings_for_combinations['zcot']['Path existence - Y-N'] + "Question: Is there a path from node 1 to node 3? "+ "There is an edge from node 1 to node 3. So there is a path from node 1 to node 3. "+ yes_answer + ex2_desc + bagSentence+ promptings_for_combinations['zcot']['Path existence - Y-N']+ "Question: Is there a path from node 2 to node 1? "+"Node 2 has only one outgoing edge to node 3. Node 3 has no outgoing edge. So there is no path from node 2 to node 1. " + no_answer
promptings_for_combinations['bag+fs+zcot']['Longest path - Path'] = stepByStepSentence + ex1_desc+ bagSentence+promptings_for_combinations['zcot']['Longest path - Path'] + "Question: What is the longest simple path from node 1 to node 3? "+"There are two simple paths from node 1 to 3. The path 1,2,3 has the length 8. The path 1,3 has the length 7. " + "Answer: 1,2,3. " + ex2_desc+ bagSentence+promptings_for_combinations['zcot']['Longest path - Path'] + "Question: What is the longest simple path from node P to node S? " +"There are two simple paths from node P to S. The path P,Q,R,S has the length 14. The path P,R,S has the length 6. "+ "Answer: P,Q,R,S. "



#Modelle
import torch
import accelerate
import json
from transformers import pipeline,AutoTokenizer, AutoModelForCausalLM
from outlines.models.transformers import Transformers
from outlines.samplers import multinomial
from outlines.generate.json import json as json_generator
import collections


model_name = "microsoft/phi-4"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_raw = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  
)
model = Transformers(model_raw, tokenizer)

schema = '''
{
  "type": "object",
  "properties": {
    "answer": {"type": "string"},
    "reasoning": {"type": "string"}
  },
  "required": ["answer","reasoning"]
}
'''

generator = json_generator(model, schema, sampler=multinomial())

schema_description = "Respond in JSON with two fields: ""answer"" (the final result) and ""reasoning"" (a brief explanation)."



# Execution
print("Start")
counter = 1

with open("resultsPMphi.csv", mode='w', newline='', encoding='utf-8') as file: 
    writer = csv.DictWriter(file, fieldnames=['Graph Category','Question Category','QuestionID', 'Question','Encoding','Prompting', 'GT', 'Antwort','Normalisiert', 'Korrekt','Reasoning','Complete prompt'], delimiter=';')
    writer.writeheader()
    encoded_graph = processGraph 
    natural_text = describe_graph(encoded_graph)
    for questionCategory, questions in questions_map.items():
        for questiondID, question in questions:
            scale_number_per_individual_question = 1
            if "<node>" in question or "<edge>" in question:
                scale_number_per_individual_question = 1 

            for scale_numer in range(scale_number_per_individual_question):
                individualQuestion, ground_truth = getQuestionForIndividualGraph(question,questiondID,encoded_graph) 

                for prompting in promptings:
                                
                    sc_prompting_activated = False
                    promptText = ""
                    parts = prompting.split("+")

                                
                    if "sc" in parts:
                        parts.remove("sc")
                        sc_prompting_activated = True

                    if prompting!= "zs":
                        if "role" in parts:
                            promptText = roles[questiondID] + promptText
                            parts.remove("role")
                        if "algo" in parts:
                            promptText = algorithms[questiondID] + promptText
                            parts.remove("algo")
                                        
                        if len(parts) > 3:
                                raise ValueError(f"Too many styles when using SC: {parts}")
                        elif len(parts) == 0:
                            completePrompt = promptText + natural_text + " Question: " + individualQuestion  + schema_description
                        elif len(parts) == 3:
                                if set(parts) == {"fs", "bag", "zcot"}:
                                    completePrompt = promptText + promptings_for_combinations["bag+fs+zcot"][questiondID] + stepByStepSentence + natural_text + bagSentence + promptings_for_combinations["zcot"][questiondID] +"Question: " + individualQuestion  + schema_description
                        elif len(parts) == 2:
                                if "fs" in parts and "bag" in parts:
                                    completePrompt = promptText + promptings_for_combinations["bag+fs"][questiondID] +natural_text+bagSentence+"Question: " + individualQuestion  + schema_description
                                elif "fs" in parts and "zcot" in parts:
                                    completePrompt = promptText + promptings_for_combinations["fs+zcot"][questiondID] + stepByStepSentence + natural_text + promptings_for_combinations["zcot"][questiondID] +"Question: " + individualQuestion  + schema_description
                                elif "bag" in parts and "zcot" in parts:
                                    completePrompt = promptText + stepByStepSentence + natural_text + bagSentence+ promptings_for_combinations["zcot"][questiondID] + "Question: " + individualQuestion  + schema_description
                        elif len(parts) == 1:
                                if parts[0] == "fs":
                                    completePrompt = promptText + promptings_for_combinations["fs"][questiondID] + natural_text + "Question: " + individualQuestion  + schema_description
                                elif parts[0] == "bag":
                                    completePrompt = promptText + natural_text + bagSentence +"Question: " + individualQuestion  + schema_description
                                elif parts[0] == "zcot":
                                    completePrompt = promptText + stepByStepSentence + natural_text + promptings_for_combinations["zcot"][questiondID] + "Question: " + individualQuestion  + schema_description
                    else: #zeroshot
                        completePrompt = natural_text + "Question: " + individualQuestion  + schema_description


                    # Prompt to LLM
                    tokens = tokenizer(completePrompt, return_tensors="pt", truncation=True, max_length=4000)
                    prompt_trimmed = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
                    reasoning = "Empty"
                    answer = "Empty"
                    if(sc_prompting_activated):
                        try:
                            results = []
                            for i in range(5):
                                res = generator(prompt_trimmed)
                                results.append(res)
                            normalized_pairs = []
                            for r in results:
                                if "answer" in r:
                                    norm = normalize_answer(r["answer"], question, encoded_graph, questiondID)
                                    if norm != "Normalizing not possible":
                                        normalized_pairs.append((norm, r.get("reasoning", "Kein Reasoning vorhanden")))
                            answers = [a for a, _ in normalized_pairs]
                            print(f"Gefundene Antworten: {answers}")
                            # Most frequent result
                            if answers:
                                most_common = collections.Counter(answers).most_common(1)
                                extracted_answer = most_common[0][0] if most_common else "None"
                                reasoning = next((r for a, r in normalized_pairs if a == extracted_answer), "No reasoning found")
                            else: 
                                extracted_answer = "Normalizing not possible"
                                reasoning ="No reasonign with SC prompting"

                        except RuntimeError :
                            extracted_answer = "Answer too long"
                            reasoning = "No reasoning"
                        except KeyError:
                            extracted_answer = "Answer does not exist"
                            reasoning = "No reasoning"
                        except json.JSONDecodeError as e:
                            extracted_answer = "Ungültiges JSON"
                            reasoning = "Parsing fehlgeschlagen"
                    else: 
                        try:
                            answer = generator(prompt_trimmed)
                            extracted_answer = answer["answer"]
                            reasoning = answer["reasoning"]
                        except RuntimeError :
                            extracted_answer = "Answer too long"
                            reasoning = "No reasoning"
                        except KeyError:
                            extracted_answer = "Answer does not exist"
                            reasoning = "No reasoning"
                        except json.JSONDecodeError as e:
                            extracted_answer = "Ungültiges JSON"
                            reasoning = "Parsing fehlgeschlagen"
                    
                    print("Counter: ", counter)
                    counter= counter+1

                    evaluation_results = evaluate_responses(encoded_graph, question, str(extracted_answer),ground_truth,questiondID)
                    for model, result in evaluation_results.items():
                        writer.writerow({
                            'Graph Category': "Process Graph",
                            'Question Category': questionCategory,
                            'QuestionID':questiondID,
                            'Question':individualQuestion,
                            'Encoding':"Process",
                            'Prompting':prompting,
                            'GT': ground_truth,
                            'Antwort': extracted_answer,
                            'Normalisiert': result['Normalisiert'],
                            'Korrekt': result['Korrekt'],
                            'Reasoning': reasoning,
                            'Complete prompt': completePrompt,
                        })
        print("Finished!")





