##### **Customer Churn Prediction System - Test Report**



##### **Group 6 - Prediction Interface / Prototype**





**1. Overview**



This document presents the testing activities conducted for the Customer Churn Prediction System.



The purpose of testing was to verify:



\- Correct integration of the trained machine learning model

\- Accuracy of churn probability predictions

\- Proper risk classification

\- Retention recommendation generation

\- Input validation functionality

\- User interface behaviour



All test cases were successfully executed using the completed Streamlit application.







**2. Testing Objectives**

The following components were tested:

User Interface

Model Integration

Churn Probability Generation 

Risk Classification

Risk Factor Identification

Retention Recommendation Generation

Input Validation

Error Handling




**3. Functional Test Cases**



**Test Case 1 - Low Risk Customer**



Input



| Attribute | Value |

|------------|---------|

| Tenure Months | 10 |

| Internet Service | Fiber Optic |

| Contract | One Year |

| Monthly Charges | 20.00 |

| Total Charges | 240.00 |

| Tech Support | Yes |



Expected Result



\- Low churn probability

\- LOW risk classification

\- Customer retention recommendation



Actual Result



| Result | Output |

|----------|---------|

| Churn Probability | 3.76% |

| Risk Level | LOW |

| Confidence | Moderate |



Status:

Passed







**Test Case 2 - Medium Risk Customer**

Input



| Attribute | Value |

|------------|---------|

| Tenure Months | 12 |

| Internet Service | DSL |

| Contract | One Year |

| Monthly Charges | 20.00 |

| Total Charges | 240.00 |



Expected Result



\- Medium churn probability

\- MEDIUM risk classification



Actual Result



| Result | Output |

|----------|---------|

| Churn Probability | 34.45% |

| Risk Level | MEDIUM |

| Confidence | Moderate |



Status:

Passed





**Test Case 3 - High Risk Customer**

Input



| Attribute | Value |

|------------|---------|

| Tenure Months | 2 |

| Internet Service | Fiber Optic |

| Contract | Month-to-month |

| Monthly Charges | 95.00 |

| Total Charges | 95.00 |

| Tech Support | No |



Expected Result



\- High churn probability

\- HIGH risk classification

\- Immediate retention recommendation



Actual Result



| Result | Output |

|----------|---------|

| Churn Probability | 74.95% |

| Risk Level | HIGH |

| Confidence | Strong |



Status:

Passed





**Test Case 4 - Very High Risk Customer**



Input



| Attribute | Value |

|------------|---------|

| Tenure Months | 0 |

| Internet Service | DSL |

| Contract | Month-to-month |

| Monthly Charges | 0.00 |

| Total Charges | 0.00 |



Expected Result



\- Very high churn probability

\- HIGH risk classification



Actual Result



| Result | Output |

|----------|---------|

| Churn Probability | 90.82% |

| Risk Level | HIGH |

| Confidence | Very Strong |



Status:

Passed





**Test Case 5 - High Risk Customer with Fiber Service**



Input



| Attribute | Value |

|------------|---------|

| Tenure Months | 10 |

| Internet Service | Fiber Optic |

| Contract | Month-to-month |

| Monthly Charges | High |

| Tech Support | No |



Expected Result



\- High churn probability

\- HIGH risk classification

\- Appropriate retention recommendation



Actual Result



| Result | Output |

|----------|---------|

| Churn Probability | 83.53% |

| Risk Level | HIGH |

| Confidence | Strong |



Status:

Passed





**4. Input Validation Testing**



Input validation was implemented to prevent invalid customer data from being processed by the machine learning model.







**Validation Test 1 - Negative Tenure Value**



Test



User attempts to enter a negative value for Tenure Months.



Expected Result



Negative values should not be accepted.



Actual Result



The Streamlit input control prevents negative values.



Status:

Passed



**Validation Test 2 - Text Input for Tenure**



Test



User attempts to enter alphabetic characters in Tenure Months.



Expected Result



Only numeric values should be accepted.



Actual Result



The Streamlit number input accepts numeric values only.



Status:

Passed





**Validation Test 3 - Total Charges Less Than Monthly Charges**



Test



Total Charges < Monthly Charges



Example:



Monthly Charges = 100



Total Charges = 50



Expected Result



System should display an error message.



Actual Result



Validation message displayed:

Status:

Passed



**Risk Classification Testing**

The risk classification mechanism was verified using multiple customer scenarios.

Probability Range	Expected Classification

0% - 30%	LOW

31% - 60%	MEDIUM

Above 60%	HIGH

Result

All tested scenarios were correctly classified.

Status:

Passed



**Recommendation Engine Testing**

The recommendation engine was tested using different risk levels.

Risk Level	Recommendation Generated

LOW	Maintain regular engagement and continue providing quality service

MEDIUM	Offer loyalty incentives and monitor customer engagement

HIGH	Provide personalized retention offer and immediate support

Result

Correct recommendation generated for each risk level.

Status:

Passed



**5. Overall Test Summary**

Test Category	                 Status



User Interface Testing	         Passed

Model Integration Testing	 Passed

Prediction Testing	         Passed

Risk Classification Testing	 Passed

Recommendation Testing	         Passed

Input Validation Testing	 Passed



**6. Conclusion**

The Customer Churn Prediction System successfully passed all functional and validation tests. The application correctly generates churn predictions, classifies customer risk levels, identifies churn drivers, provides retention recommendations, and validates user input before prediction execution.

The system is stable, user-friendly, and ready for final project submission and demonstration



