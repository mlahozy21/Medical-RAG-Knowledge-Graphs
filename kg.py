import re
import json
import os
import pickle
from typing import List, Optional, Tuple, Dict
from rdflib import Graph
from rdflib.namespace import RDFS, SKOS
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError
from json_repair import repair_json
from api import gemini
from liquid import Template
from enum import Enum

# Pydantic models for structured data
class EntityType(str, Enum):
    """Enum for valid entity types."""
    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    SYNDROME = "Syndrome"

class RelationshipType(str, Enum):
    """Enum for valid relationship types."""
    SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
    PART_OF = "http://purl.obolibrary.org/obo/RO_0002206"
    CAUSES = "http://purl.obolibrary.org/obo/RO_0003303"
    SOME_VALUES_FROM = "http://www.w3.org/2002/07/owl#someValuesFrom"
    CAUSALLY_RELATED_TO = "http://purl.obolibrary.org/obo/RO_0002610"
    HAS_MATERIAL_BASIS_IN = "http://purl.obolibrary.org/obo/RO_0004003"

class MondoEntity(BaseModel):
    mention_text: str = Field(..., description="Exact text of the entity as it appears in the query.")
    mondo_uri: Optional[str] = Field(None, description="URI from Mondo KG (e.g., http://purl.obolibrary.org/obo/MONDO_XXXXXXX).")
    mondo_label: Optional[str] = Field(None, description="Canonical label from rdfs:label or skos:prefLabel in Mondo KG.")
    type: Optional[EntityType] = Field(None, description="Inferred type of the entity (e.g., Disease, Symptom, Syndrome).")
    definition: Optional[str] = Field(None, description="Definition from skos:definition, oboInOwl:hasDefinition, or IAO_0000115, with fallback to label.")
    synonyms: List[str] = Field(default_factory=list, description="Exact or alternative synonyms from skos:exactMatch, skos:altLabel, or oboInOwl:hasExactSynonym.")
    xrefs: List[str] = Field(default_factory=list, description="Cross-references to external ontologies (e.g., ICD10, ICD9, MESH).")

    @field_validator("mondo_uri")
    @classmethod
    def validate_mondo_uri(cls, v: Optional[str]) -> Optional[str]:
        """Ensure mondo_uri follows the expected Mondo format if provided."""
        if v and not re.match(r"^http://purl\.obolibrary\.org/obo/MONDO_\d{7}$", v):
            raise ValueError("mondo_uri must be a valid Mondo URI (e.g., http://purl.obolibrary.org/obo/MONDO_XXXXXXX)")
        return v

class EntityRelationship(BaseModel):
    source_uri: str = Field(..., description="URI of the source entity in the relationship.")
    target_uri: str = Field(..., description="URI of the target entity in the relationship.")
    relation: RelationshipType = Field(..., description="Ontology predicate URI (e.g., rdfs:subClassOf, RO_0002206).")
    source_label: Optional[str] = Field(None, description="Label of the source entity from rdfs:label or skos:prefLabel.")
    target_label: Optional[str] = Field(None, description="Label of the target entity from rdfs:label or skos:prefLabel.")

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, v: str) -> str:
        """Ensure source_uri is a valid Mondo URI."""
        if not re.match(r"^http://purl\.obolibrary\.org/obo/MONDO_\d{7}$", v):
            raise ValueError("Source URI must be a valid Mondo URI (e.g., http://purl.obolibrary.org/obo/MONDO_XXXXXXX)")
        return v

    @field_validator("target_uri")
    @classmethod
    def validate_target_uri(cls, v: str) -> str:
        """Ensure target_uri is a valid URI from supported ontologies."""
        if not re.match(r"^http://(purl\.obolibrary\.org/obo/(MONDO_\d{7}|HP_\d{7}|CHR_)|identifiers\.org/hgnc/)", v):
            raise ValueError("Target URI must be a valid Mondo, HP, HGNC, or CHR URI")
        return v

class EntityContext(BaseModel):
    entities: List[MondoEntity] = Field(default_factory=list, description="List of identified entities from the query.")
    relationships: List[EntityRelationship] = Field(default_factory=list, description="List of relationships between identified entities.")

    @model_validator(mode="after")
    def validate_relationships(self):
        """Ensure source URI of relationships matches entities in the context."""
        entity_uris = {entity.mondo_uri for entity in self.entities if entity.mondo_uri}
        for rel in self.relationships:
            if rel.source_uri not in entity_uris:
                raise PydanticCustomError(
                    "relationship_source_uri_mismatch",
                    "Relationship source URI must match URIs of entities in the context"
                )
        return self

# Cache file for persistent storage
CACHE_FILE = "kg_cache.pkl"

# Load or initialize cache
def load_cache() -> Dict:
    """Loads the cache from a pickle file or initializes an empty one."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            # Validate cache structure
            if not isinstance(cache, dict) or not all(k in cache for k in ("nouns", "entity_details", "relationships", "noun_to_entity")):
                print(f"Invalid cache structure in {CACHE_FILE}, initializing empty cache")
                return {
                    "nouns": {},  # query -> List[str]
                    "entity_details": {},  # uri -> (definition, synonyms, xrefs)
                    "relationships": {},  # tuple of uris -> List[EntityRelationship]
                    "noun_to_entity": {}  # noun -> (uri, label) or None
                }
            return cache
        except Exception as e:
            print(f"Error loading cache from {CACHE_FILE}: {e}")
            return {
                "nouns": {},
                "entity_details": {},
                "relationships": {},
                "noun_to_entity": {}
            }
    return {
        "nouns": {},
        "entity_details": {},
        "relationships": {},
        "noun_to_entity": {}
    }

# Save cache to file
def save_cache(cache: Dict) -> None:
    """Saves the cache to a pickle file."""
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        print(f"Cache saved to {CACHE_FILE}")
    except Exception as e:
        print(f"Error saving cache to {CACHE_FILE}: {e}")

# Initialize cache
cache = load_cache()

# Initialize RDF graph
MONDO_FILE = "./mondo.nt"
g = Graph()
try:
    g.parse(MONDO_FILE, format="ntriples")
    print(f"Loaded KG with {len(g)} triples")
except Exception as e:
    print(f"Error loading {MONDO_FILE}: {e}")
    raise

ENTITY_LINKING_PROMPT = Template('''
You are an expert assistant for identifying medical nouns in text.
Your task is to extract all relevant nouns (focusing on diseases, syndromes, and explicitly mentioned symptoms) from the provided query.
Return a JSON object with a single field "nouns" containing a list of the exact nouns found in the text.
Do not attempt to map nouns to URIs or labels; only identify the nouns.
Return only a valid JSON object with the structure: {"nouns": [...]}
Do not return plain text or any other format. Ensure the JSON is well-formed.
Example output format:
{
  "nouns": ["diabetes", "hypertension"]
}
Now you will receive a query to process and extract the nouns.
''')

def get_entities(query: str) -> dict:
    """Extracts medical nouns from Gemini and caches the result."""
    if query in cache["nouns"]:
        print(f"Using cached nouns for query: {query}")
        return cache["nouns"][query]
    
    rendered_prompt = ENTITY_LINKING_PROMPT.render()
    content = [
        {"text": rendered_prompt},
        {"text": query}
    ]
    response = gemini.generate_content(content)
    print(f"Gemini output: {response}")
    
    # Extract the text from the response and clean the markdown block
    response_text = response.candidates[0].content.parts[0].text
    json_str = response_text.strip()[7:-3]  # Remove ```json
    parsed_response = json.loads(json_str)
    
    # Cache the result
    cache["nouns"][query] = parsed_response
    save_cache(cache)
    return parsed_response

def infer_entity_type(uri: str) -> Optional[str]:
    """Infers the type of an entity based on its URI."""
    return "Disease"  # Default for Mondo entities

def get_relationships(entity_uris: Tuple[str, ...]) -> List[EntityRelationship]:
    """Fetches relationships for entity URIs and caches the result."""
    uris_key = tuple(sorted(entity_uris))
    if uris_key in cache["relationships"]:
        print(f"Using cached relationships for URIs: {uris_key}")
        return cache["relationships"][uris_key]
    
    relationships = []
    for source_uri in entity_uris:
        sparql_query = """
        SELECT ?relation ?target ?source_label ?target_label
        WHERE {
            { <%s> ?relation ?target .
              FILTER (?relation IN (
                  <http://www.w3.org/2000/01/rdf-schema#subClassOf>,
                  <http://purl.obolibrary.org/obo/RO_0002206>,
                  <http://purl.obolibrary.org/obo/RO_0003303>,
                  <http://www.w3.org/2002/07/owl#someValuesFrom>,
                  <http://purl.obolibrary.org/obo/RO_0002610>,
                  <http://purl.obolibrary.org/obo/RO_0004003>
              ))
            }
            OPTIONAL { <%s> rdfs:label ?source_label . FILTER (lang(?source_label) = "" || lang(?source_label) = "en") }
            OPTIONAL { ?target rdfs:label ?target_label . FILTER (lang(?target_label) = "" || lang(?target_label) = "en") }
            FILTER (STRSTARTS(STR(?target), "http://purl.obolibrary.org/obo/MONDO_") ||
                    STRSTARTS(STR(?target), "http://identifiers.org/hgnc/") ||
                    STRSTARTS(STR(?target), "http://purl.obolibrary.org/obo/CHR_") ||
                    STRSTARTS(STR(?target), "http://purl.obolibrary.org/obo/HP_"))
        }
        """ % (source_uri, source_uri)
        for row in g.query(sparql_query):
            try:
                relationships.append(EntityRelationship(
                    source_uri=source_uri,
                    target_uri=str(row.target),
                    relation=str(row.relation),
                    source_label=str(row.source_label) if row.source_label else None,
                    target_label=str(row.target_label) if row.target_label else None
                ))
            except ValueError as e:
                print(f"Skipping invalid relationship for {source_uri}: {e}")
                continue
    
    if not relationships:
        print(f"No relationships found for URIs: {uris_key}")
    
    cache["relationships"][uris_key] = relationships
    save_cache(cache)
    return relationships

def fetch_entity_details(uri: str) -> Tuple[Optional[str], List[str], List[str]]:
    """Fetches and caches entity details (definition, synonyms, xrefs)."""
    if uri in cache["entity_details"]:
        print(f"Using cached entity details for {uri}")
        return cache["entity_details"][uri]
    
    # Label (rdfs:label, skos:prefLabel, or oboInOwl:hasExactSynonym)
    label = None
    sparql_query = """
    SELECT ?label
    WHERE {
        { <%s> rdfs:label ?label . }
        UNION { <%s> <http://www.w3.org/2004/02/skos/core#prefLabel> ?label . }
        UNION { <%s> <http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> ?label . }
        FILTER (lang(?label) = "" || lang(?label) = "en")
    }
    ORDER BY STRLEN(?label)
    LIMIT 1
    """ % (uri, uri, uri)
    for row in g.query(sparql_query):
        label = str(row.label)
    print(f"Fetched label for {uri}: {label}")

    # Definition (IAO_0000115, skos:definition, or oboInOwl:hasDefinition)
    definition = None
    sparql_query = """
    SELECT ?definition
    WHERE {
        { <%s> <http://purl.obolibrary.org/obo/IAO_0000115> ?definition . }
        UNION { <%s> <http://www.w3.org/2004/02/skos/core#definition> ?definition . }
        UNION { <%s> <http://www.geneontology.org/formats/oboInOwl#hasDefinition> ?definition . }
        FILTER (lang(?definition) = "" || lang(?definition) = "en")
    }
    LIMIT 1
    """ % (uri, uri, uri)
    for row in g.query(sparql_query):
        definition = str(row.definition)
    print(f"Fetched definition for {uri}: {definition}")

    # Validate definition
    if definition and "obsolete" in definition.lower():
        print(f"Warning: Definition for {uri} contains 'obsolete' ('{definition}'), falling back to label")
        definition = label
    elif not definition and label:
        print(f"No definition found for {uri}, using label as fallback")
        definition = label

    # Synonyms
    synonyms = []
    sparql_query = """
    SELECT ?label
    WHERE {
        { <%s> <http://www.w3.org/2004/02/skos/core#exactMatch> ?synonym .
          { ?synonym rdfs:label ?label . }
          UNION { ?synonym <http://www.w3.org/2004/02/skos/core#prefLabel> ?label . }
          UNION { ?synonym <http://www.w3.org/2004/02/skos/core#altLabel> ?label . }
        }
        UNION { <%s> <http://www.w3.org/2004/02/skos/core#altLabel> ?label . }
        UNION { <%s> <http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> ?label . }
        UNION { <%s> <http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym> ?label . }
        FILTER (lang(?label) = "" || lang(?label) = "en")
        FILTER (!STRSTARTS(str(?label), "http://"))
    }
    LIMIT 10
    """ % (uri, uri, uri, uri)
    for row in g.query(sparql_query):
        synonym_label = str(row.label)
        print(f"Found synonym for {uri}: {synonym_label}")
        if synonym_label and not synonym_label.startswith("http://"):
            synonyms.append(synonym_label)
    
    # Deduplicate synonyms and exclude the main label
    synonyms = list(set(synonyms) - {label})

    # Cross-references
    xrefs = []
    sparql_query = """
    SELECT ?xref
    WHERE {
        <%s> <http://www.geneontology.org/formats/oboInOwl#hasDbXref> ?xref .
    }
    LIMIT 10
    """ % uri
    for row in g.query(sparql_query):
        xref = str(row.xref)
        if xref.startswith(("ICD10", "ICD9", "MESH")):
            xrefs.append(xref)

    # Cache the result
    cache["entity_details"][uri] = (definition, synonyms, xrefs)
    save_cache(cache)
    return definition, synonyms, xrefs

def format_context(context: EntityContext) -> str:
    """Formats the entity context into a readable string."""
    lines = []
    for entity in context.entities:
        line = f"{entity.mention_text} (Label: {entity.mondo_label}, Definition: {entity.definition or 'Not available'})"
        if entity.synonyms:
            filtered_synonyms = list(dict.fromkeys(entity.synonyms))[:5]
            line += f". It is also known as {', '.join(filtered_synonyms)}"
        if entity.xrefs:
            line += f". Relevant medical codes include {', '.join(entity.xrefs)}"
        lines.append(line)
    
    else:
        for rel in context.relationships:
            rel_type = rel.relation.split("#")[-1] if "#" in rel.relation else rel.relation
            lines.append(f"Relationship: {rel.source_label or rel.source_uri} {rel_type} {rel.target_label or rel.target_uri}")
    
    return "\n".join(lines)

def verify_entities(nouns: List[str]) -> None:
    """Runs a diagnostic query to check if nouns exist in the KG."""
    for noun in nouns:
        sparql_query = """
        SELECT DISTINCT ?entity ?label
        WHERE {
            {
                ?entity rdfs:label ?label .
                FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
            } UNION {
                ?entity <http://www.w3.org/2004/02/skos/core#prefLabel> ?label .
                FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
            } UNION {
                ?entity <http://www.w3.org/2004/02/skos/core#altLabel> ?label .
                FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
            } UNION {
                ?entity <http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> ?label .
                FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
            } UNION {
                ?entity <http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym> ?label .
                FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
            }
            FILTER (STRSTARTS(STR(?entity), "http://purl.obolibrary.org/obo/MONDO_"))
            FILTER (lang(?label) = "" || lang(?label) = "en")
        }
        LIMIT 5
        """ % (noun, noun, noun, noun, noun)
        print(f"Diagnostic query for '{noun}':")
        for row in g.query(sparql_query):
            print(f" - Found: URI={row.entity}, Label={row.label}")

def get_entity_context(query: str) -> str:
    """Generates entity context from a query by searching nouns in the Mondo KG."""
    # Extract nouns using Gemini
    linked_nouns = get_entities(query)
    
    if not linked_nouns["nouns"]:
        return "No medical nouns identified in the query."
    
    print(f"Searching nouns in KG: {linked_nouns['nouns']}")
    
    entities = []
    for noun in linked_nouns["nouns"]:
        # Check if noun mapping is cached
        if noun in cache["noun_to_entity"]:
            print(f"Using cached noun mapping for '{noun}'")
            result = cache["noun_to_entity"][noun]
            if result is None:
                print(f"No entity found in cache for '{noun}'")
                with open("unmatched_nouns.txt", "a") as f:
                    f.write(f"{noun}\n")
                continue
            uri, label = result
        else:
            # Exact match query
            sparql_query = """
            SELECT DISTINCT ?entity ?label
            WHERE {
                {
                    ?entity rdfs:label ?label .
                    FILTER (LCASE(str(?label)) = LCASE("%s"))
                } UNION {
                    ?entity <http://www.w3.org/2004/02/skos/core#prefLabel> ?label .
                    FILTER (LCASE(str(?label)) = LCASE("%s"))
                } UNION {
                    ?entity <http://www.w3.org/2004/02/skos/core#altLabel> ?label .
                    FILTER (LCASE(str(?label)) = LCASE("%s"))
                } UNION {
                    ?entity <http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> ?label .
                    FILTER (LCASE(str(?label)) = LCASE("%s"))
                } UNION {
                    ?entity <http://www.w3.org/2004/02/skos/core#hasRelatedSynonym> ?label .
                    FILTER (LCASE(str(?label)) = LCASE("%s"))
                } UNION {
                    ?entity <http://www.w3.org/2004/02/skos/core#exactMatch> ?synonym .
                    ?synonym rdfs:label ?label .
                    FILTER (LCASE(str(?label)) = LCASE("%s"))
                }
                FILTER (STRSTARTS(STR(?entity), "http://purl.obolibrary.org/obo/MONDO_"))
                FILTER (lang(?label) = "" || lang(?label) = "en")
            }
            ORDER BY STRLEN(?label)
            LIMIT 1
            """ % (noun, noun, noun, noun, noun, noun)
            
            result = g.query(sparql_query)
            found = False
            exact_match = True
            for row in result:
                print(f"Found entity for '{noun}': URI={row.entity}, Label={row.label}")
                uri = str(row.entity)
                label = str(row.label)
                cache["noun_to_entity"][noun] = (uri, label)
                found = True
            
            if not found:
                # Partial match query
                sparql_query = """
                SELECT DISTINCT ?entity ?label
                WHERE {
                    {
                        ?entity rdfs:label ?label .
                        FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
                    } UNION {
                        ?entity <http://www.w3.org/2004/02/skos/core#prefLabel> ?label .
                        FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
                    } UNION {
                        ?entity <http://www.w3.org/2004/02/skos/core#altLabel> ?label .
                        FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
                    } UNION {
                        ?entity <http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> ?label .
                        FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
                    } UNION {
                        ?entity <http://www.w3.org/2004/02/skos/core#hasRelatedSynonym> ?label .
                        FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
                    } UNION {
                        ?entity <http://www.w3.org/2004/02/skos/core#exactMatch> ?synonym .
                        ?synonym rdfs:label ?label .
                        FILTER (CONTAINS(LCASE(str(?label)), LCASE("%s")))
                    }
                    FILTER (STRSTARTS(STR(?entity), "http://purl.obolibrary.org/obo/MONDO_"))
                    FILTER (lang(?label) = "" || lang(?label) = "en")
                    OPTIONAL { ?entity <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?parent }
                    BIND(IF(BOUND(?parent), 1, 0) AS ?subclass_count)
                }
                ORDER BY ?subclass_count STRLEN(?label)
                LIMIT 1
                """ % (noun, noun, noun, noun, noun, noun)
                
                result = g.query(sparql_query)
                exact_match = False
                for row in result:
                    uri = str(row.entity)
                    label = str(row.label)
                    if "obsolete" in label.lower():
                        print(f"Skipping obsolete entity for '{noun}': URI={uri}, Label={label}")
                        continue
                    print(f"Found entity (partial match) for '{noun}': URI={uri}, Label={label}")
                    cache["noun_to_entity"][noun] = (uri, label)
                    found = True
                
                if not found or ("obsolete" in label.lower() and not exact_match):
                    print(f"No valid entity found in KG for '{noun}'")
                    cache["noun_to_entity"][noun] = None
                    with open("unmatched_nouns.txt", "a") as f:
                        f.write(f"{noun}\n")
            
            save_cache(cache)
        
        if cache["noun_to_entity"].get(noun) is not None:
            uri, label = cache["noun_to_entity"][noun]
            entity = MondoEntity(
                mention_text=noun,
                mondo_uri=uri,
                mondo_label=label,
                type=infer_entity_type(uri)
            )
            entities.append(entity)
    
    if not entities:
        return "No matching entities found in the Mondo Knowledge Graph."
    
    for entity in entities:
        if entity.mondo_uri and not entity.definition:
            definition, synonyms, xrefs = fetch_entity_details(entity.mondo_uri)
            entity.definition = definition
            entity.synonyms = synonyms
            entity.xrefs = xrefs
            if not entity.type:
                entity.type = infer_entity_type(entity.mondo_uri)

    entity_uris = [entity.mondo_uri for entity in entities if entity.mondo_uri]
    relationships = get_relationships(tuple(entity_uris))

    context = EntityContext(entities=entities, relationships=relationships)
    context_text = format_context(context)
    return context_text

if __name__ == "__main__":
    sample_query = "Is diabetes related with hypertension?"
    context = get_entity_context(sample_query)
    print("Generated Context:")
    print(context)
